package com.live2d.facecapture

import android.os.SystemClock
import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.CALIBRATION_DURATION_MS
import com.example.commondata.FaceLandmarkIndices
import com.example.commondata.HeadPose
import com.example.commondata.MIN_CALIBRATION_FRAMES
import com.example.commondata.Point3D


/**
 * Face Capture Pipeline - 完整的面部追踪管线
 */
class FaceCapturePipeline {
    // ============ 组件 ============

    // BlendShape 特征提取器
    val extractor = BlendShapeExtractor()


    // ============ 姿态基线 ============

    private var baselinePitch: Float = 0f
    private var baselineYaw: Float = 0f
    private var baselineRoll: Float = 0f
    private var isPoseCalibrated: Boolean = false

    // 用于自动校准的累积数据
    //
    // 校准窗口现在按"时长 + 最小帧数"双门限触发，而不是写死的帧数 —— 这样无论手机
    // 性能 / 实际帧率高低，校准都基于大致相同的真实时间窗口，不会在 release 包飙到
    // 60 FPS 时只用 ~330ms 就草草结束。
    private var calibrationFrameCount: Int = 0
    private var calibrationStartTimestampMs: Long = 0L
    private var calibrationPitchSum: Float = 0f
    private var calibrationYawSum: Float = 0f
    private var calibrationRollSum: Float = 0f

    // 用于点位平均的累积数据（只累积 landmarks，因为 FaceKeyPoints 结构复杂）
    private var calibrationLandmarksAccumulator: MutableList<List<Point3D>> = mutableListOf()

    /**
     * 处理结果
     */
    data class ProcessResult(
        val blendShapes: ARKitBlendShapes,
        val relativePose: HeadPose,  // 相对于基线的姿态角（用于 Live2D 和 UI）
        val rawPose: HeadPose,       // 原始姿态角（用于调试）
        val isCalibrating: Boolean = false              // 是否正在校准中
    )

    /**
     * 是否正在校准期
     * 综合判断：姿态校准 + 眼睛校准 + 嘴部校准 + 微笑校准
     */
    val isCalibrating: Boolean
        get() = !isPoseCalibrated || extractor.isCalibrating

    /**
     * 处理一帧 MediaPipe 数据
     *
     * @param landmarks MediaPipe FaceMesh 输出的 478 个点
     * @param imageWidth 图像宽度
     * @param imageHeight 图像高度
     * @param rawPitch 原始俯仰角（度）
     * @param rawYaw 原始偏航角（度）
     * @param rawRoll 原始翻滚角（度）
     * @return 处理结果，包含 BlendShape 和相对姿态角
     */
    fun processWithPose(
        landmarks: List<Point3D>,
        imageWidth: Int = 0,
        imageHeight: Int = 0,
        rawPitch: Float,
        rawYaw: Float,
        rawRoll: Float
    ): ProcessResult {
        // 更新人脸距离估计（用于动态调整透视补偿系数）
        updateFaceDistanceFromLandmarks(landmarks, imageWidth, imageHeight)

        // 3. 提取关键点（先提取，因为校准需要用到）
        val keyPoints = FaceLandmarkIndices.extractKeyPoints(landmarks, imageWidth, imageHeight)

        // 1. 自动校准姿态基线（按"时长 + 最小帧数"窗口取平均）
        if (!isPoseCalibrated) {
            if (calibrationFrameCount == 0) {
                calibrationStartTimestampMs = SystemClock.uptimeMillis()
            }
            calibrationFrameCount++
            calibrationPitchSum += rawPitch
            calibrationYawSum += rawYaw
            calibrationRollSum += rawRoll

            // 累积点位数据
            calibrationLandmarksAccumulator.add(landmarks)

            val elapsedMs = SystemClock.uptimeMillis() - calibrationStartTimestampMs
            val windowReady = elapsedMs >= CALIBRATION_DURATION_MS &&
                calibrationFrameCount >= MIN_CALIBRATION_FRAMES

            if (windowReady) {
                // 注意：除数必须是实际收集到的帧数，而不是任何常量 —— 在变长窗口下，
                // 不同帧率 / 不同设备拿到的样本数都会不同。
                val n = calibrationFrameCount.toFloat()
                baselinePitch = calibrationPitchSum / n
                baselineYaw = calibrationYawSum / n
                baselineRoll = calibrationRollSum / n
                isPoseCalibrated = true

                // 使用累积的点位数据计算平均值进行校准
                calibrateWithAccumulatedData()

                Log.i("FaceCapturePipeline",
                    "✅ Auto-calibrated: pose=(%.1f, %.1f, %.1f), %d frames over %d ms".format(
                        baselinePitch, baselineYaw, baselineRoll,
                        calibrationFrameCount, elapsedMs
                    ))
            }
        }

        // 2. 计算相对姿态角（相对于校准基线）
        val relativePitch = if (isPoseCalibrated) rawPitch - baselinePitch else rawPitch
        val relativeYaw = if (isPoseCalibrated) rawYaw - baselineYaw else rawYaw
        val relativeRoll = if (isPoseCalibrated) rawRoll - baselineRoll else rawRoll

        val relativePose = HeadPose(relativePitch, relativeYaw, relativeRoll)
        val rawPose = HeadPose(rawPitch, rawYaw, rawRoll)

        // 4. 提取 BlendShape
        val rawBlendShapes = extractor.extractFullBlendShapes(keyPoints, landmarks, rawPose)

        return ProcessResult(
            blendShapes = rawBlendShapes,
            relativePose = relativePose,
            rawPose = rawPose,
            isCalibrating = isCalibrating
        )
    }

    /**
     * 使用累积的点位数据进行校准（取平均值）
     * 在自动校准完成时调用
     */
    private fun calibrateWithAccumulatedData() {
        if (calibrationLandmarksAccumulator.isEmpty()) {
            Log.w("FaceCapturePipeline", "No accumulated data for calibration")
            return
        }

        val accumulatedCount = calibrationLandmarksAccumulator.size

        // 计算平均 landmarks
        val avgLandmarks = averageLandmarks(calibrationLandmarksAccumulator)

        // 从平均 landmarks 提取 keyPoints
        val avgKeyPoints = FaceLandmarkIndices.extractKeyPoints(avgLandmarks)

        // 使用平均值进行校准
        extractor.calibrate(avgKeyPoints, avgLandmarks)

        // 设置嘴部补偿的校准姿态（使用原始姿态角的平均值）
        extractor.setCalibrationPose(baselinePitch, baselineYaw, baselineRoll)

        // 清空累积数据
        calibrationLandmarksAccumulator.clear()

        Log.i("FaceCapturePipeline",
            "✅ Expression calibrated with $accumulatedCount frames average landmarks")
    }

    /**
     * 计算 landmarks 的平均值
     */
    private fun averageLandmarks(landmarksList: List<List<Point3D>>): List<Point3D> {
        if (landmarksList.isEmpty()) {
            return emptyList()
        }
        if (landmarksList.size == 1) {
            return landmarksList[0]
        }

        val n = landmarksList.size.toFloat()
        val size = landmarksList[0].size

        return (0 until size).map { i ->
            val avgX = landmarksList.sumOf { it[i].x.toDouble() }.toFloat() / n
            val avgY = landmarksList.sumOf { it[i].y.toDouble() }.toFloat() / n
            val avgZ = landmarksList.sumOf { it[i].z.toDouble() }.toFloat() / n
            Point3D(avgX, avgY, avgZ)
        }
    }

    /**
     * 从 landmarks 估算人脸大小，更新动态补偿系数
     */
    private fun updateFaceDistanceFromLandmarks(
        landmarks: List<Point3D>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        if (landmarks.isEmpty()) return

        // 从 landmarks 计算人脸边界框
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE

        for (point in landmarks) {
            if (point.x < minX) minX = point.x
            if (point.x > maxX) maxX = point.x
            if (point.y < minY) minY = point.y
            if (point.y > maxY) maxY = point.y
        }

        // 检查是否有有效的边界框
        if (minX >= maxX || minY >= maxY) return

        val faceWidth: Float
        val faceHeight: Float

        // 判断坐标是归一化的还是像素坐标
        // 如果 maxX <= 1.0 且 maxY <= 1.0，说明是归一化坐标
        if (maxX <= 1.5f && maxY <= 1.5f && minX >= -0.5f && minY >= -0.5f) {
            // 归一化坐标，直接使用
            faceWidth = maxX - minX
            faceHeight = maxY - minY
        } else if (imageWidth > 0 && imageHeight > 0) {
            // 像素坐标，需要归一化
            faceWidth = (maxX - minX) / imageWidth
            faceHeight = (maxY - minY) / imageHeight
        } else {
            // 无法确定，跳过
            return
        }

        // 更新 SimpleEyeExtractor 的距离估计
        extractor.updateFaceDistanceEstimateNormalized(faceWidth, faceHeight)
    }

    fun setCameraFov(fov: Float) {
        extractor.setCameraFov(fov)
    }

    /**
     * 重置所有状态
     */
    fun reset() {
        // 重置姿态基线
        baselinePitch = 0f
        baselineYaw = 0f
        baselineRoll = 0f
        isPoseCalibrated = false
        calibrationFrameCount = 0
        calibrationStartTimestampMs = 0L
        calibrationPitchSum = 0f
        calibrationYawSum = 0f
        calibrationRollSum = 0f

        // 重置累积数据
        calibrationLandmarksAccumulator.clear()

        // 重置表情提取器
        extractor.reset()

        Log.i("FaceCapturePipeline", "🔄 Pipeline reset")
    }
}