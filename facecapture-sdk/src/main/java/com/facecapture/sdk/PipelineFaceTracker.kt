package com.facecapture.sdk

import android.content.Context
import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.DetectionBox
import com.example.commondata.FaceVisualizationData
import com.example.commondata.HeadPoseData
import com.example.commondata.Point3D
import com.example.mediapipefacelandmark.Camera2Manager
import com.example.mediapipefacelandmark.MediaPipeFaceDetector
import com.example.mediapipefacelandmark.YuvFrameData
import com.live2d.facecapture.BlinkCurveProcessor
import com.live2d.facecapture.FaceCapturePipeline
import com.live2d.facecapture.TrackingConfidenceDetector
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Pipeline-based Face Tracker with TRUE parallel processing stages.
 *
 * This is the **direct / advanced** API of the face-capture AAR. It owns its
 * own [Camera2Manager], so callers only need to construct it, wire the
 * callbacks they care about and call [start] / [stop].
 *
 * For a stricter, IO-agnostic facade see [FaceCaptureEngine].
 *
 * Architecture (Two-Stage Parallel Pipeline):
 *
 * The key insight is that Face Detection and Landmark Detection use SEPARATE
 * MNN Sessions, so they CAN run in parallel.
 *
 * - Stage 1: Face Detection (~4ms)      - finds face bounding box
 * - Stage 2: Landmark Detection (~24ms) - uses PREVIOUS frame's bbox
 * - Stage 3: BlendShape Processing (~10ms) - uses current landmarks
 *
 * This achieves: max(4ms Det, 24ms Landmark) = 24ms = ~41 FPS theoretical.
 *
 * Note: Using previous frame's bbox is acceptable because face position
 * changes very little between consecutive frames at 30+ FPS.
 */
class PipelineFaceTracker(private val context: Context) {

    companion object {
        private const val TAG = "PipelineFaceTracker"

        // Pipeline configuration
        private const val DETECTION_CHANNEL_CAPACITY = 2
        private const val LANDMARK_CHANNEL_CAPACITY = 2
        private const val PROCESSING_CHANNEL_CAPACITY = 4
        private const val FPS_UPDATE_INTERVAL = 1000L

        // Performance profiling
        private const val ENABLE_PERF_PROFILING = true
        private const val PERF_LOG_INTERVAL = 60
    }

    // Performance stats
    private data class PerfStats(
        var total: Double = 0.0,
        var min: Double = Double.MAX_VALUE,
        var max: Double = 0.0,
        var count: Int = 0,
    ) {
        fun add(ms: Double) {
            total += ms
            min = minOf(min, ms)
            max = maxOf(max, ms)
            count++
        }
        fun avg(): Double = if (count > 0) total / count else 0.0
        fun reset() { total = 0.0; min = Double.MAX_VALUE; max = 0.0; count = 0 }
    }

    private val perfDetectionTotal = PerfStats()
    private val perfLandmarkTotal = PerfStats()
    private val perfProcessingTotal = PerfStats()
    private val perfBlendShape = PerfStats()
    private val perfApplyToLive2D = PerfStats()
    private var perfFrameCount = 0
    private val perfLock = Any()

    private var detectionRunCount = 0
    private var detectionSkipCount = 0
    private var redetectTriggerCount = 0

    // Core components
    private val camera2Manager = Camera2Manager(context)
    private val faceDetector = MediaPipeFaceDetector(context)
    val faceCapturePipeline: FaceCapturePipeline = FaceCapturePipeline()
    val blinkCurveProcessor = BlinkCurveProcessor()

    @Volatile private var lastFaceBox: MediaPipeFaceDetector.FaceBox? = null
    private val faceBoxLock = Any()

    @OptIn(ExperimentalCoroutinesApi::class)
    private val faceDetectionDispatcher = Dispatchers.Default.limitedParallelism(1)

    @OptIn(ExperimentalCoroutinesApi::class)
    private val landmarkDispatcher = Dispatchers.Default.limitedParallelism(1)

    @OptIn(ExperimentalCoroutinesApi::class)
    private val processingDispatcher = Dispatchers.Default.limitedParallelism(1)

    private val frameChannel = Channel<FrameData>(
        capacity = DETECTION_CHANNEL_CAPACITY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val landmarkFrameChannel = Channel<FrameData>(
        capacity = LANDMARK_CHANNEL_CAPACITY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    private val landmarkChannel = Channel<LandmarkData>(
        capacity = PROCESSING_CHANNEL_CAPACITY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    private val pipelineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val isRunning = AtomicBoolean(false)
    private var lastHeadPose = HeadPoseData()

    private val trackingConfidenceDetector by lazy { TrackingConfidenceDetector() }

    // 注意：姿态校准现在由 FaceCapturePipeline 内部管理
    // 注意：角度归一化（用于驱动 Live2D 等下游 avatar）由调用方负责，
    //       PipelineFaceTracker 只输出原始相对姿态（已减去校准基线，单位：度）。

    private var frameCount = AtomicLong(0)
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0f

    private var landmarkOutputCount = AtomicLong(0)
    private var lastLandmarkFpsTime = System.currentTimeMillis()

    private var cameraFrameCount = AtomicLong(0)
    private var lastCameraFpsTime = System.currentTimeMillis()

    // Callbacks
    var onCameraFrameUpdate: ((ByteBuffer, Int, Int) -> Unit)? = null
    var onBlendShapesUpdate: ((ARKitBlendShapes) -> Unit)? = null
    var onHeadPoseUpdate: ((HeadPoseData) -> Unit)? = null
    var onVisualizationUpdate: ((FaceVisualizationData) -> Unit)? = null
    var onFpsUpdate: ((Float) -> Unit)? = null
    var onCalibrationStatusUpdate: ((Boolean) -> Unit)? = null

    /**
     * 原子组合回调：BlendShape + HeadPose 同帧到达，专门给下游 avatar 驱动用。
     * 与 [onBlendShapesUpdate] / [onHeadPoseUpdate] 的区别：后两者面向 UI 调试，可独立订阅；
     * 这个回调保证同一帧的两类数据原子配对，避免下游 avatar 出现 BS/姿态错位。
     */
    var onFaceCaptureFrame: ((ARKitBlendShapes, HeadPoseData) -> Unit)? = null

    // ============================================================================
    // Internal data classes
    // ============================================================================

    private data class FrameData(
        val frameId: Long,
        val buffer: ByteBuffer,
        val width: Int,
        val height: Int,
        val timestamp: Long = System.currentTimeMillis(),
    )

    private data class YuvFrameDataInternal(
        val frameId: Long,
        val yuvData: YuvFrameData,
        val timestamp: Long = System.currentTimeMillis(),
    )

    private data class LandmarkData(
        val frameId: Long,
        val landmarks: List<Point3D>,
        val width: Int,
        val height: Int,
        val timestamp: Long,
        val headPose: Triple<Float, Float, Float>?,
    )

    /** Use direct YUV processing (recommended; avoids Kotlin-side YUV->RGB conversion). */
    var useDirectYuvMode: Boolean = true

    private val yuvFrameChannel = Channel<YuvFrameDataInternal>(
        capacity = DETECTION_CHANNEL_CAPACITY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    private val yuvLandmarkFrameChannel = Channel<YuvFrameDataInternal>(
        capacity = LANDMARK_CHANNEL_CAPACITY,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

    // ============================================================================
    // Lifecycle
    // ============================================================================

    fun initialize(): Boolean {
        Log.i(TAG, "Initializing pipeline face tracker...")

        if (!faceDetector.initialize(numThreads = 2)) {
            Log.e(TAG, "Failed to initialize face detector")
            return false
        }

        val dumpDir = context.cacheDir.absolutePath
        faceDetector.setDumpDirectory(dumpDir)

        Log.i(TAG, "Pipeline face tracker initialized with parallel stages")
        return true
    }

    fun start() {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "Pipeline already running")
            return
        }

        Log.i(TAG, "Starting parallel pipeline face tracker (YUV direct mode: $useDirectYuvMode)...")

        if (useDirectYuvMode) {
            startYuvFaceDetectionStage()
            startYuvLandmarkStage()
        } else {
            startFaceDetectionStage()
            startLandmarkStage()
        }
        startProcessingStage()

        var frameId = 0L

        if (useDirectYuvMode) {
            camera2Manager.useDirectYuvMode = true
            camera2Manager.enablePreviewInYuvMode = true

            camera2Manager.setFrameCallback { frameData, width, height ->
                onCameraFrameUpdate?.invoke(frameData.duplicate(), width, height)
            }

            camera2Manager.setYuvFrameCallback { yuvData ->
                cameraFrameCount.incrementAndGet()
                val now = System.currentTimeMillis()
                val elapsed = now - lastCameraFpsTime
                if (elapsed >= 5000) {
                    val cameraFps = cameraFrameCount.get() * 1000f / elapsed
                    Log.i(TAG, "Camera input rate: %.1f FPS (YUV direct mode)".format(cameraFps))
                    cameraFrameCount.set(0)
                    lastCameraFpsTime = now
                }

                val data = YuvFrameDataInternal(frameId = frameId++, yuvData = yuvData)
                yuvFrameChannel.trySend(data)
                yuvLandmarkFrameChannel.trySend(data.copy())
            }
        } else {
            camera2Manager.useDirectYuvMode = false
            camera2Manager.setFrameCallback { frameData, width, height ->
                onCameraFrameUpdate?.invoke(frameData.duplicate(), width, height)

                cameraFrameCount.incrementAndGet()
                val now = System.currentTimeMillis()
                val elapsed = now - lastCameraFpsTime
                if (elapsed >= 5000) {
                    val cameraFps = cameraFrameCount.get() * 1000f / elapsed
                    Log.i(TAG, "Camera input rate: %.1f FPS".format(cameraFps))
                    cameraFrameCount.set(0)
                    lastCameraFpsTime = now
                }

                val buffer = copyBuffer(frameData)
                val data = FrameData(frameId = frameId++, buffer = buffer, width = width, height = height)
                frameChannel.trySend(data)
                landmarkFrameChannel.trySend(data.copy(buffer = copyBuffer(frameData)))
            }
        }

        camera2Manager.startCamera()

        pipelineScope.launch(Dispatchers.Default) {
            delay(500)
            val fov = camera2Manager.cameraVerticalFovDegrees
            faceDetector.setCameraFov(fov)
            faceCapturePipeline.setCameraFov(fov)
            Log.i(TAG, "Camera FOV set to face detector: $fov degrees")
        }

        lastFpsTime = System.currentTimeMillis()
        frameCount.set(0)
    }

    fun stop() {
        if (!isRunning.getAndSet(false)) return
        Log.i(TAG, "Stopping pipeline face tracker...")
        camera2Manager.stopCamera()
        frameChannel.close()
        landmarkFrameChannel.close()
        landmarkChannel.close()
    }

    fun release() {
        stop()
        faceDetector.release()
        pipelineScope.cancel()
        Log.i(TAG, "Pipeline face tracker released")
    }

    fun hasPermission(): Boolean = camera2Manager.hasPermission()
    fun isTracking(): Boolean = isRunning.get()
    fun getLastHeadPose(): HeadPoseData = lastHeadPose
    fun getCurrentFps(): Float = currentFps

    /**
     * Reset adaptive baseline for BlendShape extraction.
     * 这会同时重置：
     * - 姿态基线（FaceCapturePipeline 内部管理）
     * - 表情基线
     *
     * 注意：下游 avatar 适配器（如 Live2DAvatarAdapter）的归一化器需要由
     *      调用方在收到本方法的副作用后另行 reset，本类不再耦合任何 avatar 相关状态。
     */
    fun resetAdaptiveBaseline() {
        Log.i(TAG, "Resetting all baselines...")
        faceCapturePipeline.reset()
        faceDetector.resetSmoothing()
        blinkCurveProcessor.reset()
        trackingConfidenceDetector.reset()
        synchronized(faceBoxLock) { lastFaceBox = null }
        Log.i(TAG, "✅ All baselines reset, auto-calibration will start on next frames")
    }

    @Deprecated("Use resetAdaptiveBaseline() instead")
    fun recalibrateHeadPose() {
        Log.i(TAG, "Resetting head pose calibration via pipeline...")
        faceCapturePipeline.reset()
    }

    // ============================================================================
    // Pipeline stages
    // ============================================================================

    private fun startFaceDetectionStage() {
        pipelineScope.launch(faceDetectionDispatcher) {
            Log.i(TAG, "Face detection stage started")
            for (frame in frameChannel) {
                if (!isRunning.get()) break
                try {
                    val startTime = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val box = faceDetector.detectFaceOnly(frame.buffer, frame.width, frame.height)
                    if (box != null) {
                        synchronized(faceBoxLock) { lastFaceBox = box }
                    }
                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            perfDetectionTotal.add((System.nanoTime() - startTime) / 1_000_000.0)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Face detection error", e)
                }
            }
            Log.i(TAG, "Face detection stage stopped")
        }
    }

    private fun startLandmarkStage() {
        pipelineScope.launch(landmarkDispatcher) {
            Log.i(TAG, "Landmark detection stage started")
            for (frame in landmarkFrameChannel) {
                if (!isRunning.get()) break
                try {
                    val box = synchronized(faceBoxLock) { lastFaceBox } ?: continue
                    val startTime = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val landmarks = faceDetector.detectLandmarksWithBox(
                        frame.buffer, frame.width, frame.height, box,
                    )
                    if (landmarks != null && landmarks.isNotEmpty()) {
                        val headPose = faceDetector.estimateHeadPose(
                            landmarks, frame.width, frame.height,
                        )
                        val landmarkData = LandmarkData(
                            frameId = frame.frameId,
                            landmarks = landmarks,
                            width = frame.width,
                            height = frame.height,
                            timestamp = frame.timestamp,
                            headPose = headPose,
                        )
                        landmarkOutputCount.incrementAndGet()
                        val now = System.currentTimeMillis()
                        val elapsed = now - lastLandmarkFpsTime
                        if (elapsed >= 5000) {
                            val landmarkFps = landmarkOutputCount.get() * 1000f / elapsed
                            Log.i(TAG, "Landmark output rate: %.1f FPS".format(landmarkFps))
                            landmarkOutputCount.set(0)
                            lastLandmarkFpsTime = now
                        }
                        landmarkChannel.send(landmarkData)
                    }
                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            perfLandmarkTotal.add((System.nanoTime() - startTime) / 1_000_000.0)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Landmark detection error", e)
                }
            }
            Log.i(TAG, "Landmark detection stage stopped")
        }
    }

    private fun startYuvFaceDetectionStage() {
        pipelineScope.launch(faceDetectionDispatcher) {
            Log.i(TAG, "YUV Face detection stage started")
            for (frame in yuvFrameChannel) {
                if (!isRunning.get()) break
                try {
                    val startTime = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val yuvData = frame.yuvData
                    val box = faceDetector.detectFaceFromYuv(
                        yuvData.yBuffer, yuvData.uBuffer, yuvData.vBuffer,
                        yuvData.yRowStride, yuvData.uvRowStride, yuvData.uvPixelStride,
                        yuvData.width, yuvData.height, yuvData.rotation,
                    )
                    if (box != null) {
                        synchronized(faceBoxLock) { lastFaceBox = box }
                    }
                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            detectionRunCount++
                            perfDetectionTotal.add((System.nanoTime() - startTime) / 1_000_000.0)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "YUV Face detection error", e)
                }
            }
            Log.i(TAG, "YUV Face detection stage stopped")
        }
    }

    private fun startYuvLandmarkStage() {
        pipelineScope.launch(landmarkDispatcher) {
            Log.i(TAG, "YUV Landmark detection stage started")
            for (frame in yuvLandmarkFrameChannel) {
                if (!isRunning.get()) break
                try {
                    val box = synchronized(faceBoxLock) { lastFaceBox }
                    if (box == null) {
                        synchronized(perfLock) { detectionSkipCount++ }
                        continue
                    }
                    val startTime = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val yuvData = frame.yuvData
                    val landmarkResult = faceDetector.detectLandmarksFromYuvWithScore(
                        yuvData.yBuffer, yuvData.uBuffer, yuvData.vBuffer,
                        yuvData.yRowStride, yuvData.uvRowStride, yuvData.uvPixelStride,
                        yuvData.rotatedWidth, yuvData.rotatedHeight, yuvData.rotation, box,
                    )
                    if (landmarkResult != null && landmarkResult.landmarks.isNotEmpty()) {
                        val landmarks = landmarkResult.landmarks
                        val presenceScore = landmarkResult.presenceScore

                        val trackingStatus = trackingConfidenceDetector.updateAndCheck(
                            currentLandmarks = landmarks,
                            presenceScore = presenceScore,
                        )
                        if (trackingStatus.shouldRedetect) {
                            Log.d(TAG, "Tracking confidence low, triggering re-detection: ${trackingStatus.reason}")
                            synchronized(faceBoxLock) { lastFaceBox = null }
                            synchronized(perfLock) { redetectTriggerCount++ }
                            trackingConfidenceDetector.onDetectionPerformed()
                        }

                        val headPose = faceDetector.estimateHeadPose(
                            landmarks, yuvData.rotatedWidth, yuvData.rotatedHeight,
                        )
                        val landmarkData = LandmarkData(
                            frameId = frame.frameId,
                            landmarks = landmarks,
                            width = yuvData.rotatedWidth,
                            height = yuvData.rotatedHeight,
                            timestamp = frame.timestamp,
                            headPose = headPose,
                        )

                        landmarkOutputCount.incrementAndGet()
                        val now = System.currentTimeMillis()
                        val elapsed = now - lastLandmarkFpsTime
                        if (elapsed >= 5000) {
                            val landmarkFps = landmarkOutputCount.get() * 1000f / elapsed
                            Log.i(TAG, "YUV Landmark output rate: %.1f FPS".format(landmarkFps))
                            landmarkOutputCount.set(0)
                            lastLandmarkFpsTime = now
                        }
                        landmarkChannel.send(landmarkData)
                    }
                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            perfLandmarkTotal.add((System.nanoTime() - startTime) / 1_000_000.0)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "YUV Landmark detection error", e)
                }
            }
            Log.i(TAG, "YUV Landmark detection stage stopped")
        }
    }

    private fun startProcessingStage() {
        pipelineScope.launch(processingDispatcher) {
            Log.i(TAG, "Processing stage started")
            for (data in landmarkChannel) {
                if (!isRunning.get()) break
                try {
                    val stageStart = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val headPose = data.headPose
                    val bsStart = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val rawPitch = headPose?.first ?: 0f
                    val rawYaw = headPose?.second ?: 0f
                    val rawRoll = headPose?.third ?: 0f

                    val result = faceCapturePipeline.processWithPose(
                        data.landmarks, data.width, data.height,
                        rawPitch, rawYaw, rawRoll,
                    )
                    val blendShapes = result.blendShapes

                    onCalibrationStatusUpdate?.invoke(result.isCalibrating)

                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            perfBlendShape.add((System.nanoTime() - bsStart) / 1_000_000.0)
                        }
                    }

                    val relativePose = result.relativePose
                    val headPoseData = HeadPoseData(
                        relativePose.pitch, relativePose.yaw, relativePose.roll,
                    )
                    lastHeadPose = headPoseData

                    updateFps()

                    blinkCurveProcessor.updateFrameRate(60f)

                    val processedBlendShapes = if (blinkCurveProcessor.isEnabled) {
                        val (leftBlink, rightBlink) = blinkCurveProcessor.process(
                            blendShapes.eyeBlinkLeft,
                            blendShapes.eyeBlinkRight,
                            System.currentTimeMillis(),
                        )
                        blendShapes.copy(
                            eyeBlinkLeft = leftBlink,
                            eyeBlinkRight = rightBlink,
                        )
                    } else {
                        blendShapes
                    }

                    val applyStart = if (ENABLE_PERF_PROFILING) System.nanoTime() else 0L
                    val currentBlendShapes = processedBlendShapes
                    val currentHeadPoseData = headPoseData
                    val currentLandmarks = data.landmarks
                    val currentWidth = data.width
                    val currentHeight = data.height

                    pipelineScope.launch(Dispatchers.Default) {
                        onFaceCaptureFrame?.invoke(currentBlendShapes, currentHeadPoseData)

                        val detectionBox = calculateBoundingBox(currentLandmarks)
                        val visualizationData = FaceVisualizationData(
                            landmarks = currentLandmarks,
                            detectionBox = detectionBox,
                            headPose = currentHeadPoseData,
                            imageWidth = currentWidth,
                            imageHeight = currentHeight,
                        )

                        onBlendShapesUpdate?.invoke(currentBlendShapes)
                        onVisualizationUpdate?.invoke(visualizationData)
                        onHeadPoseUpdate?.invoke(currentHeadPoseData)
                    }
                    if (ENABLE_PERF_PROFILING) {
                        synchronized(perfLock) {
                            perfApplyToLive2D.add((System.nanoTime() - applyStart) / 1_000_000.0)
                            perfProcessingTotal.add((System.nanoTime() - stageStart) / 1_000_000.0)
                            perfFrameCount++
                            if (perfFrameCount % PERF_LOG_INTERVAL == 0) {
                                logPerfStats()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Processing error", e)
                }
            }
            Log.i(TAG, "Processing stage stopped")
        }
    }

    // ============================================================================
    // Helpers
    // ============================================================================

    private fun logPerfStats() {
        val detAvg = perfDetectionTotal.avg()
        val lmAvg = perfLandmarkTotal.avg()
        val procAvg = perfProcessingTotal.avg()

        val theoreticalMaxFps = if (lmAvg > 0) 1000.0 / lmAvg else 0.0

        val totalFrames = detectionRunCount + detectionSkipCount
        val detectionRate = if (totalFrames > 0) 100.0 * detectionRunCount / totalFrames else 0.0
        val skipRate = if (totalFrames > 0) 100.0 * detectionSkipCount / totalFrames else 0.0

        Log.i(TAG, "=== Parallel Pipeline Perf (avg of $PERF_LOG_INTERVAL frames) ===")
        Log.i(TAG, "  Face Detection: %.2f ms (runs in parallel)".format(detAvg))
        Log.i(TAG, "    Detection runs: $detectionRunCount (%.1f%%)".format(detectionRate))
        Log.i(TAG, "    Skipped (reuse box): $detectionSkipCount (%.1f%%)".format(skipRate))
        Log.i(TAG, "    Re-detect triggers: $redetectTriggerCount (low confidence)")
        Log.i(TAG, "  Landmark Detection: %.2f ms (runs in parallel)".format(lmAvg))
        Log.i(TAG, "  Processing: %.2f ms".format(procAvg))
        Log.i(TAG, "    BlendShape: %.2f ms (%.1f%%)".format(
            perfBlendShape.avg(), if (procAvg > 0) 100.0 * perfBlendShape.avg() / procAvg else 0.0))
        Log.i(TAG, "    Apply+UI:   %.2f ms (%.1f%%)".format(
            perfApplyToLive2D.avg(), if (procAvg > 0) 100.0 * perfApplyToLive2D.avg() / procAvg else 0.0))
        Log.i(TAG, "  Bottleneck: Landmark (%.2f ms)".format(lmAvg))
        Log.i(TAG, "  Theoretical max FPS (1000/landmark_ms): %.0f FPS".format(theoreticalMaxFps))
        Log.i(TAG, "  Actual output FPS (UI display): %.1f FPS".format(currentFps))

        perfDetectionTotal.reset()
        perfLandmarkTotal.reset()
        perfProcessingTotal.reset()
        perfBlendShape.reset()
        perfApplyToLive2D.reset()
        detectionRunCount = 0
        detectionSkipCount = 0
        redetectTriggerCount = 0
    }

    private fun copyBuffer(source: ByteBuffer): ByteBuffer {
        source.rewind()
        val copy = ByteBuffer.allocateDirect(source.remaining())
        copy.put(source)
        copy.rewind()
        source.rewind()
        return copy
    }

    private fun calculateBoundingBox(landmarks: List<Point3D>): DetectionBox {
        if (landmarks.isEmpty()) return DetectionBox(0f, 0f, 1f, 1f)

        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE

        for (landmark in landmarks) {
            minX = minOf(minX, landmark.x)
            minY = minOf(minY, landmark.y)
            maxX = maxOf(maxX, landmark.x)
            maxY = maxOf(maxY, landmark.y)
        }

        val padding = 0.02f
        return DetectionBox(
            x1 = (minX - padding).coerceAtLeast(0f),
            y1 = (minY - padding).coerceAtLeast(0f),
            x2 = (maxX + padding).coerceAtMost(1f),
            y2 = (maxY + padding).coerceAtMost(1f),
            score = 1.0f,
        )
    }

    private fun updateFps() {
        frameCount.incrementAndGet()
        val now = System.currentTimeMillis()
        val elapsed = now - lastFpsTime

        if (elapsed >= FPS_UPDATE_INTERVAL) {
            val finalCount = frameCount.getAndSet(0)
            val finalElapsed = now - lastFpsTime
            lastFpsTime = now

            currentFps = finalCount * 1000f / finalElapsed
            Log.d(TAG, "UI FPS update: count=$finalCount, elapsed=${finalElapsed}ms, fps=%.1f".format(currentFps))
            onFpsUpdate?.invoke(currentFps)
        }
    }
}
