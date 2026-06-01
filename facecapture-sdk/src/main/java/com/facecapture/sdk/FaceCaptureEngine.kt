package com.facecapture.sdk

import android.content.Context
import android.media.Image
import com.example.commondata.DetectionBox
import com.example.commondata.HeadPose
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.Point3D
import java.nio.ByteBuffer

/**
 * # FaceCaptureEngine
 *
 * Public, stable entry point exposed by the **facecapture-sdk** AAR.
 *
 * Pipeline contract (one input frame → one output result):
 *
 * ```
 *  CameraFrame (YUV_420_888)
 *        │
 *        ▼  MNN inference (MediaPipe FaceLandmarker, 478 pts)
 *        ▼  OpenSeeFace-style post-processing
 *        ▼  ARKit-52 BlendShape extraction
 *        ▼  Adaptive head-pose normalization
 *        ▼  Blink curve & temporal smoothing
 *        ▼
 *  FaceCaptureResult  (landmarks + headPose + blendshapes)
 * ```
 *
 * The engine is thread-safe. Frames are dispatched onto an internal
 * coroutine pipeline that drops oldest frames under back-pressure, so
 * callers may push at arbitrary rates without blocking the camera thread.
 *
 * ## Typical usage
 *
 * ```kotlin
 * val engine = FaceCaptureEngine.create()
 * engine.initialize(context)
 * engine.setResultListener { result ->
 *     // --- Raw perception layer (MediaPipe FaceLandmarker on MNN) ---
 *     val rawLandmarks = result.landmarks          // 478 normalized 3D points
 *     val imgW         = result.imageWidth         // size of the frame those
 *     val imgH         = result.imageHeight        //   landmarks live in
 *     val faceBox      = result.detectionBox       // tight bbox over landmarks
 *     val presence     = result.presenceScore      // model confidence [0,1]
 *
 *     // --- Post-processed layer (OpenSeeFace-style stack) ---
 *     val blendShapes  = result.blendShapes        // 52 ARKit BlendShapes, smoothed
 *     val rawPose      = result.rawHeadPose        // raw pitch/yaw/roll
 *     val relPose      = result.relativeHeadPose   // baseline-subtracted
 * }
 *
 * // From Camera2 / CameraX ImageAnalysis:
 * imageProxy.image?.let { engine.pushFrame(it, rotationDegrees = 270) }
 *
 * // On teardown:
 * engine.release()
 * ```
 *
 * > **Implementation status**: this interface is shipped as the public
 * > contract of the AAR. The reference implementation lives in
 * > [com.facecapture.sdk.internal.FaceCaptureEngineImpl] and is built by
 * > extracting `PipelineFaceTracker` from the demo `app/` module. See the
 * > module README §"Migrating PipelineFaceTracker into the SDK" for the
 * > step-by-step refactor.
 */
interface FaceCaptureEngine {

    /**
     * Load MNN models and allocate native resources.
     *
     * @param context  any Context; only `applicationContext` is retained.
     * @param config   optional tunables; defaults are sensible for live capture.
     * @return `true` on success; `false` if model loading failed (check logcat).
     */
    fun initialize(context: Context, config: FaceCaptureConfig = FaceCaptureConfig()): Boolean

    /** Register a callback invoked once per processed frame, on a worker thread. */
    fun setResultListener(listener: (FaceCaptureResult) -> Unit)

    /**
     * Push a raw YUV_420_888 frame for inference.
     *
     * Caller retains ownership of the buffers; the engine copies what it needs
     * before returning. Safe to call from any thread.
     */
    fun pushFrame(frame: CameraFrame)

    /**
     * Convenience overload that accepts an [android.media.Image] directly
     * (e.g. from CameraX `ImageAnalysis.Analyzer` or Camera2 `ImageReader`).
     *
     * @param rotationDegrees clockwise rotation needed to make the image upright
     *                        (0 / 90 / 180 / 270). Pass `imageInfo.rotationDegrees`
     *                        from CameraX, or compute from the sensor orientation.
     */
    fun pushFrame(image: Image, rotationDegrees: Int)

    /** Discard the current pose / expression / blink baselines and recalibrate. */
    fun recalibrate()

    /** Camera vertical field-of-view (degrees). Improves head-pose accuracy. */
    fun setCameraVerticalFov(degrees: Float)

    /** Release native + coroutine resources. The instance is unusable afterwards. */
    fun release()

    companion object {
        /**
         * Factory. Returns the default reference implementation.
         *
         * v0.1.0: the engine drives its own Camera2 capture internally —
         * after [initialize], cast the returned instance to
         * `com.facecapture.sdk.internal.FaceCaptureEngineImpl` and call
         * `startInternalCamera()` / `stopInternalCamera()`. The
         * [pushFrame] entry points are reserved for v0.2.0.
         *
         * If you need the full power-user surface today, instantiate
         * [com.facecapture.sdk.PipelineFaceTracker] directly — it is part of
         * the public AAR API.
         */
        @JvmStatic
        fun create(): FaceCaptureEngine =
            com.facecapture.sdk.internal.FaceCaptureEngineImpl()
    }
}

/**
 * Tunable configuration passed once at [FaceCaptureEngine.initialize].
 *
 * @property modelAssetDir       Sub-directory inside the AAR `assets/` folder
 *                               that holds the `.mnn` model files.
 *                               `null` ⇒ use built-in defaults.
 * @property externalModelDir    Absolute filesystem path overriding `assets/`.
 *                               Useful when models are downloaded at runtime.
 * @property numThreads          MNN thread count for inference (1..4).
 * @property enableBlinkCurve    Apply the natural-blink curve post-processor
 *                               (recommended for Live2D / VTuber use).
 * @property calibrationFrames   Number of initial frames used to learn the
 *                               neutral baseline (head pose + expression).
 * @property smoothing           Temporal smoothing filter type.
 */
data class FaceCaptureConfig(
    val modelAssetDir: String? = null,
    val externalModelDir: String? = null,
    val numThreads: Int = 2,
    val enableBlinkCurve: Boolean = true,
    val calibrationFrames: Int = 30,
    val smoothing: SmoothingMode = SmoothingMode.TIME_BASED_EMA,
)

enum class SmoothingMode { NONE, EMA, TIME_BASED_EMA, ONE_EURO }

/**
 * Frame input. Either build one manually from camera buffers or use the
 * [FaceCaptureEngine.pushFrame] overload that accepts [android.media.Image].
 *
 * Layout matches `ImageFormat.YUV_420_888`.
 */
data class CameraFrame(
    val y: ByteBuffer,
    val u: ByteBuffer,
    val v: ByteBuffer,
    val yRowStride: Int,
    val uvRowStride: Int,
    val uvPixelStride: Int,
    val width: Int,
    val height: Int,
    /** Clockwise rotation (degrees) to make the frame upright. */
    val rotationDegrees: Int,
    val timestampNanos: Long = System.nanoTime(),
)

/**
 * Per-frame inference + post-processing result.
 *
 * The SDK deliberately exposes **both** layers of the pipeline so that
 * integrators are free to pick what they need:
 *
 * - [landmarks] / [imageWidth] / [imageHeight] / [detectionBox] / [presenceScore]
 *   come straight out of the MediaPipe-FaceLandmarker MNN model
 *   (Detection → Landmark). They are the *raw* perception output —
 *   useful for custom rigs, debug overlays, or running your own post-processing.
 *
 * - [rawHeadPose] / [relativeHeadPose] / [blendShapes] are the *post-processed*
 *   outputs of the OpenSeeFace-style stack (calibration, smoothing,
 *   blink-curve, ARKit-52 solver). These are what you typically forward to
 *   a Live2D / VRM / VTuber rig.
 *
 * All fields belong to the **same frame** ([frameId]), guaranteed atomic
 * by the engine — landmarks will never go out of sync with blendshapes.
 *
 * @property frameId            Monotonically increasing frame id.
 * @property landmarks          Raw 478 MediaPipe FaceMesh points. Coordinates
 *                              are normalized to `[0, 1]` over
 *                              ([imageWidth] x [imageHeight]); `z` is the
 *                              relative depth produced by the model.
 * @property imageWidth         Width  (px) of the image the landmarks were
 *                              produced on (after camera rotation).
 * @property imageHeight        Height (px) of the image the landmarks were
 *                              produced on (after camera rotation).
 * @property detectionBox       Tight bounding box around [landmarks]
 *                              (normalized), useful for cropping / overlays.
 * @property rawHeadPose        Head pose computed directly from landmarks
 *                              (degrees, no baseline subtraction).
 * @property relativeHeadPose   Head pose with the calibrated baseline
 *                              subtracted; this is what you typically feed
 *                              to a downstream avatar/Live2D rig.
 * @property blendShapes        Full set of 52 ARKit BlendShape values,
 *                              already smoothed + blink-curve-processed.
 * @property presenceScore      Face-presence confidence in `[0, 1]`,
 *                              reported by the Landmark model.
 * @property isCalibrating      `true` while still inside the calibration window.
 * @property fps                Rolling inference FPS reported by the engine.
 */
data class FaceCaptureResult(
    val frameId: Long,
    val landmarks: List<Point3D>,
    val imageWidth: Int,
    val imageHeight: Int,
    val detectionBox: DetectionBox,
    val rawHeadPose: HeadPose,
    val relativeHeadPose: HeadPose,
    val blendShapes: ARKitBlendShapes,
    val presenceScore: Float,
    val isCalibrating: Boolean,
    val fps: Float,
)
