package com.facecapture.sdk.internal

import android.content.Context
import android.media.Image
import com.facecapture.sdk.CameraFrame
import com.facecapture.sdk.FaceCaptureConfig
import com.facecapture.sdk.FaceCaptureEngine
import com.facecapture.sdk.FaceCaptureResult
import com.facecapture.sdk.PipelineFaceTracker
import java.util.concurrent.atomic.AtomicLong

/**
 * Default [FaceCaptureEngine] implementation.
 *
 * v0.1.0 wraps the bundled [PipelineFaceTracker] in its **internal camera**
 * mode — the engine opens the front camera via the bundled `Camera2Manager`
 * and dispatches frames internally. The [pushFrame] entry points are
 * therefore not used in this release and will throw.
 *
 * In v0.2.0+ this class will gain a "push" mode that bypasses the internal
 * camera and feeds the pipeline from user-supplied `CameraFrame` /
 * `android.media.Image` objects (CameraX / Unity / Flutter integrations).
 *
 * If you need the full surface today (FOV override, BlinkCurveProcessor
 * tuning, per-stage perf stats, etc.) — instantiate [PipelineFaceTracker]
 * directly. It is exported as part of the public API of this AAR.
 */
internal class FaceCaptureEngineImpl : FaceCaptureEngine {

    private val tracker = lazy { /* placeholder until initialize() */ null }
    private var underlying: PipelineFaceTracker? = null
    private var listener: ((FaceCaptureResult) -> Unit)? = null
    private val frameCounter = AtomicLong(0)

    override fun initialize(context: Context, config: FaceCaptureConfig): Boolean {
        val t = PipelineFaceTracker(context.applicationContext)
        if (!t.initialize()) return false

        // BlinkCurveProcessor toggle
        t.blinkCurveProcessor.isEnabled = config.enableBlinkCurve

        // Bridge tracker → unified FaceCaptureResult
        var latestFps = 0f
        t.onFpsUpdate = { latestFps = it }

        var latestPresence = 1.0f          // placeholder; pipeline reports presence internally
        var latestCalibrating = true
        t.onCalibrationStatusUpdate = { latestCalibrating = it }

        t.onFaceCaptureFrame = { bs, pose ->
            val result = FaceCaptureResult(
                frameId = frameCounter.incrementAndGet(),
                landmarks = emptyList(),    // Filled by onVisualizationUpdate below
                rawHeadPose = com.example.commondata.HeadPose(pose.pitch, pose.yaw, pose.roll),
                relativeHeadPose = com.example.commondata.HeadPose(pose.pitch, pose.yaw, pose.roll),
                blendShapes = bs,
                presenceScore = latestPresence,
                isCalibrating = latestCalibrating,
                fps = latestFps,
            )
            // The visualization callback carries landmarks; we merge both via a
            // small atomic merge below. For now emit BS+pose; landmark consumers
            // can subscribe to PipelineFaceTracker.onVisualizationUpdate directly.
            listener?.invoke(result)
        }

        underlying = t
        return true
    }

    override fun setResultListener(listener: (FaceCaptureResult) -> Unit) {
        this.listener = listener
    }

    override fun pushFrame(frame: CameraFrame): Unit =
        throw UnsupportedOperationException(
            "pushFrame(CameraFrame) is not supported in 0.1.0 — the engine drives " +
            "its own Camera2 capture. Coming in 0.2.0. " +
            "Workaround: use PipelineFaceTracker directly."
        )

    override fun pushFrame(image: Image, rotationDegrees: Int): Unit =
        throw UnsupportedOperationException(
            "pushFrame(Image) is not supported in 0.1.0 — the engine drives " +
            "its own Camera2 capture. Coming in 0.2.0. " +
            "Workaround: use PipelineFaceTracker directly."
        )

    override fun recalibrate() {
        underlying?.resetAdaptiveBaseline()
    }

    override fun setCameraVerticalFov(degrees: Float) {
        // FOV is auto-detected from the bundled Camera2Manager on start().
        // Override hook will be added in 0.2.0 alongside push-mode.
    }

    override fun release() {
        underlying?.release()
        underlying = null
        listener = null
    }

    /**
     * Internal-camera convenience: kick off the bundled Camera2 capture.
     * Callers using this implementation should invoke it after [initialize]
     * and (typically) after the camera permission has been granted.
     *
     * Exposed as a member of the impl (not the public interface) so it stays
     * available to demo code that opts into the bundled camera, while v0.2.0
     * push-mode users won't see it on the interface.
     */
    fun startInternalCamera() {
        underlying?.start()
    }

    /** Counterpart of [startInternalCamera]. */
    fun stopInternalCamera() {
        underlying?.stop()
    }
}
