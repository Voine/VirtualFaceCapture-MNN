package com.example.virtualfacecapture

import android.util.Log
import com.example.commondata.HeadPoseData
import com.live2d.demo.JniBridgeJava
import com.example.commondata.ARKitBlendShapes

/**
 * Live2DAvatarAdapter
 * ----------------------------------------------------------------------
 * 把【面捕模块的中性输出】适配到【Live2D Cubism 模型参数】。
 *
 * 这个类是 PipelineFaceTracker → JniBridgeJava 之间唯一的桥梁：
 * - 上游订阅 [PipelineFaceTracker.onFaceCaptureFrame] 拿到原子配对的
 *   `(ARKitBlendShapes, HeadPoseData)`，转发给 [onFrame]。
 * - 本类内部维护 3 轴自适应角度归一化器，把"原始相对欧拉角（度）"映射到
 *   Live2D 期望的 `[-1, 1]` 区间，然后调用 [JniBridgeJava] 写入模型参数。
 *
 * 这样 PipelineFaceTracker / facecapture-sdk 可以完全脱离 Live2D，
 * 任何想换 avatar 框架（Unity humanoid / VRM / Spine ...）的下游
 * 只需要再实现一个同形态的 adapter，面捕侧零改动。
 */
class Live2DAvatarAdapter {

    companion object {
        private const val TAG = "Live2DAvatarAdapter"
    }

    /**
     * Adaptive angle normalizer for asymmetric ranges.
     * Tracks min/max values separately for positive and negative directions
     * and normalizes to [-1, 1] based on observed range.
     *
     * 仅本类内部使用 —— 把原始度数映射到 Live2D 参数空间。
     */
    private class AdaptiveAngleNormalizer(
        private var defaultNegativeRange: Float = 30f,
        private var defaultPositiveRange: Float = 30f,
        private val minRange: Float = 5f,
        private val adaptationRate: Float = 0.1f,
        private val usePercentile: Boolean = true,
    ) {
        private var observedNegativeMax: Float = 0f
        private var observedPositiveMax: Float = 0f
        private var percentileNegative: Float = 0f
        private var percentilePositive: Float = 0f
        private var sampleCount: Int = 0

        fun normalize(value: Float): Float {
            updateRanges(value)
            val negRange = getEffectiveNegativeRange()
            val posRange = getEffectivePositiveRange()
            return if (value < 0) {
                (value / negRange).coerceIn(-1f, 0f)
            } else {
                (value / posRange).coerceIn(0f, 1f)
            }
        }

        private fun updateRanges(value: Float) {
            sampleCount++
            if (usePercentile) {
                val alpha = if (sampleCount < 100) 0.05f else 0.01f
                if (value < 0) {
                    val absValue = -value
                    if (absValue > percentileNegative) {
                        percentileNegative += (absValue - percentileNegative) * adaptationRate
                    } else {
                        percentileNegative += (absValue - percentileNegative) * alpha * 0.1f
                    }
                } else {
                    if (value > percentilePositive) {
                        percentilePositive += (value - percentilePositive) * adaptationRate
                    } else {
                        percentilePositive += (value - percentilePositive) * alpha * 0.1f
                    }
                }
            }
            if (value < 0) {
                observedNegativeMax = maxOf(observedNegativeMax, -value)
            } else {
                observedPositiveMax = maxOf(observedPositiveMax, value)
            }
        }

        private fun getEffectiveNegativeRange(): Float {
            val observed = if (usePercentile) percentileNegative else observedNegativeMax
            val blendFactor = (sampleCount / 100f).coerceIn(0f, 1f)
            val range = defaultNegativeRange * (1 - blendFactor) + observed * blendFactor
            return maxOf(range, minRange)
        }

        private fun getEffectivePositiveRange(): Float {
            val observed = if (usePercentile) percentilePositive else observedPositiveMax
            val blendFactor = (sampleCount / 100f).coerceIn(0f, 1f)
            val range = defaultPositiveRange * (1 - blendFactor) + observed * blendFactor
            return maxOf(range, minRange)
        }

        fun reset() {
            observedNegativeMax = 0f
            observedPositiveMax = 0f
            percentileNegative = 0f
            percentilePositive = 0f
            sampleCount = 0
        }
    }

    // Pitch: asymmetric (low head ~20°, up head ~40°)
    private val pitchNormalizer = AdaptiveAngleNormalizer(
        defaultNegativeRange = 20f,
        defaultPositiveRange = 40f,
        minRange = 10f,
    )

    // Yaw: roughly symmetric (~45° each direction)
    private val yawNormalizer = AdaptiveAngleNormalizer(
        defaultNegativeRange = 45f,
        defaultPositiveRange = 45f,
        minRange = 15f,
    )

    // Roll: roughly symmetric (~30° each direction)
    private val rollNormalizer = AdaptiveAngleNormalizer(
        defaultNegativeRange = 30f,
        defaultPositiveRange = 30f,
        minRange = 10f,
    )

    /**
     * 主入口：每帧调用一次。
     * @param blendShapes 已经过眨眼曲线/平滑处理的 52 项 ARKit BlendShape
     * @param headPose    已减去校准基线的相对欧拉角（度）
     */
    fun onFrame(blendShapes: ARKitBlendShapes, headPose: HeadPoseData) {
        try {
            // -------- 眼睛开合（从 blink [0,1] 翻转为 open [0,1]） --------
            val eyeLOpen = 1.0f - blendShapes.eyeBlinkLeft
            val eyeROpen = 1.0f - blendShapes.eyeBlinkRight
            JniBridgeJava.nativeProjectEyeLOpen(eyeLOpen)
            JniBridgeJava.nativeProjectEyeROpen(eyeROpen)

            // -------- 嘴部 --------
            val mouthOpen = blendShapes.jawOpen
            val mouthForm =
                (blendShapes.mouthSmileLeft + blendShapes.mouthSmileRight) / 2.0f -
                (blendShapes.mouthFrownLeft + blendShapes.mouthFrownRight) / 2.0f
            JniBridgeJava.nativeProjectMouthOpenY(mouthOpen)
            JniBridgeJava.nativeProjectMouthForm(mouthForm)

            // -------- 头部欧拉角（度 → [-1, 1] 自适应归一化） --------
            val normalizedPitch = pitchNormalizer.normalize(headPose.pitch)
            val normalizedYaw = yawNormalizer.normalize(headPose.yaw)
            val normalizedRoll = rollNormalizer.normalize(headPose.roll)
            // 注意：符号约定按现有模型校准过；换模型若方向相反在此调整正负号
            JniBridgeJava.nativeModelEulerParameters(-normalizedPitch, -normalizedYaw, normalizedRoll)

            // -------- 视线方向（ARKit eyeLookIn/Out/Up/Down → Live2D X/Y） --------
            // X: + = 向画面右侧看
            val leftEyeX = blendShapes.eyeLookOutLeft - blendShapes.eyeLookInLeft
            val rightEyeX = blendShapes.eyeLookInRight - blendShapes.eyeLookOutRight
            val eyeBallX = (leftEyeX + rightEyeX) / 2.0f
            // Y: + = 向上看
            val leftEyeY = blendShapes.eyeLookUpLeft - blendShapes.eyeLookDownLeft
            val rightEyeY = blendShapes.eyeLookUpRight - blendShapes.eyeLookDownRight
            val eyeBallY = (leftEyeY + rightEyeY) / 2.0f
            JniBridgeJava.nativeProjectEyeBallX(eyeBallX)
            JniBridgeJava.nativeProjectEyeBallY(eyeBallY)
        } catch (e: Exception) {
            Log.e(TAG, "Error applying blendshapes to Live2D", e)
        }
    }

    /**
     * 重置自适应归一化器。
     * 应在 [PipelineFaceTracker.resetAdaptiveBaseline] 触发时同步调用，
     * 这样 avatar 的归一化窗口和面捕基线保持一致。
     */
    fun reset() {
        pitchNormalizer.reset()
        yawNormalizer.reset()
        rollNormalizer.reset()
    }
}
