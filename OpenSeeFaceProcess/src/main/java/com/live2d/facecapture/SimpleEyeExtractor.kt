package com.live2d.facecapture

import android.os.SystemClock
import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.CALIBRATION_DURATION_MS
import com.example.commondata.EyePoints
import com.example.commondata.HeadPose
import com.example.commondata.MIN_CALIBRATION_FRAMES
import kotlin.math.*

/**
 * 简化的眼睛 BlendShape 提取器
 * 支持自适应阈值校准和头部姿态补偿
 */
class SimpleEyeExtractor {

    companion object {
        // ============ 默认阈值（用于未校准时）============
        // 这些是保守的默认值，实际阈值会通过校准自动调整
        private const val DEFAULT_EYE_CLOSED_RATIO = 0.10f
        private const val DEFAULT_EYE_OPEN_RATIO = 0.18f
        private const val DEFAULT_EYE_WIDE_RATIO = 0.26f

        // ============ 眼睛灵敏度控制 ============

        /**
         * 眼睛灵敏度系数
         *
         * 这个参数控制眼睛开合检测的灵敏度：
         * - 1.0 = 默认灵敏度
         * - > 1.0 = 更灵敏（轻微眨眼就会被放大）
         * - < 1.0 = 不那么灵敏（需要更大幅度的眨眼）
         *
         * 推荐范围: 0.5 ~ 2.0
         */
        var eyeSensitivity: Float = 1.0f
            set(value) {
                field = value.coerceIn(0.3f, 3.0f)
            }
    }
    
    /**
     * 校准时的头部姿态
     * 用于计算"相对于校准时刻的透视变化量"
     */
    private var calibrationPitch: Float = 0f
    private var calibrationYaw: Float = 0f
    private var calibrationFaceRatio: Float = 0.5f

    /**
     * 眼睛校准数据类
     * 用于存储校准期间收集的统计信息
     */
    data class EyeCalibrationData(
        // 累积统计
        var sumEAR: Float = 0f,
        var sumSquaredEAR: Float = 0f,
        var minEAR: Float = Float.MAX_VALUE,
        var maxEAR: Float = Float.MIN_VALUE,
        var frameCount: Int = 0,
        // 校准窗口起始时间（uptimeMillis），用于按时长触发结束
        var startTimestampMs: Long = 0L,

        // 校准结果
        var meanEAR: Float = 0f,
        var stdEAR: Float = 0f,
        var calibratedClosedRatio: Float = DEFAULT_EYE_CLOSED_RATIO,
        var calibratedOpenRatio: Float = DEFAULT_EYE_OPEN_RATIO,
        var calibratedWideRatio: Float = DEFAULT_EYE_WIDE_RATIO,

        var isCalibrated: Boolean = false
    ) {
        /**
         * 添加一帧 EAR 数据
         */
        fun addFrame(ear: Float) {
            if (frameCount == 0) {
                startTimestampMs = SystemClock.uptimeMillis()
            }
            sumEAR += ear
            sumSquaredEAR += ear * ear
            minEAR = min(minEAR, ear)
            maxEAR = max(maxEAR, ear)
            frameCount++
        }

        /**
         * 校准窗口是否已经收集够样本（时长 + 最小帧数 双门限）。
         * 见 [CALIBRATION_DURATION_MS] / [MIN_CALIBRATION_FRAMES] 的说明。
         */
        fun isWindowReady(): Boolean {
            if (frameCount < MIN_CALIBRATION_FRAMES) return false
            val elapsed = SystemClock.uptimeMillis() - startTimestampMs
            return elapsed >= CALIBRATION_DURATION_MS
        }

        /**
         * 完成校准，计算阈值
         */
        fun finishCalibration() {
            if (frameCount < 10) {
                Log.w("EyeCalibration", "Too few frames for calibration: $frameCount")
                return
            }

            // 计算统计值
            meanEAR = sumEAR / frameCount
            val variance = (sumSquaredEAR / frameCount) - (meanEAR * meanEAR)
            stdEAR = sqrt(max(0f, variance))

            val observedRange = maxEAR - minEAR

            // openRatio = 正常睁眼的 EAR
            // 使用 mean - 0.5*std 作为上界（留出一些余量）
            // 或者使用 minEAR + 一些余量（观察到的最小睁眼 EAR）
            val observedOpenRatio = if (observedRange > 0.02f) {
                // 有足够变化，说明可能包含了眨眼，使用 max - 10%
                maxEAR - 0.1f * observedRange
            } else {
                // 变化太小，用户可能一直睁眼，使用 mean - 1*std
                meanEAR - stdEAR
            }

            // closedRatio = 闭眼的 EAR
            // 关键：如果校准期间没有观察到眨眼，我们无法准确知道闭眼时的 EAR
            // 因此使用保守的默认值，或者基于 openRatio 推断
            val inferredClosedRatio = if (observedRange > 0.03f && minEAR < meanEAR - 2 * stdEAR) {
                // 观察到了明显的眨眼，可以使用观察到的最小值
                minEAR + 0.1f * observedRange
            } else {
                // 没有观察到眨眼，基于 openRatio 推断
                // 假设闭眼时 EAR 下降约 35-40%（从 0.18 降到 0.11 左右）
                // 使用 0.6 倍而不是 0.5 倍，让完全闭眼更容易触发
                observedOpenRatio * 0.6f
            }

            // 应用约束
            calibratedOpenRatio = observedOpenRatio.coerceIn(0.12f, 0.35f)
            calibratedClosedRatio = inferredClosedRatio.coerceIn(0.05f, calibratedOpenRatio - 0.03f)
            calibratedWideRatio = calibratedOpenRatio + (calibratedOpenRatio - calibratedClosedRatio) * 0.8f

            // 安全检查：确保范围不会太窄
            val finalRange = calibratedOpenRatio - calibratedClosedRatio
            if (finalRange < 0.04f) {
                Log.w("EyeCalibration", "Calibrated range too narrow ($finalRange), using defaults")
                calibratedClosedRatio = DEFAULT_EYE_CLOSED_RATIO
                calibratedOpenRatio = DEFAULT_EYE_OPEN_RATIO
                calibratedWideRatio = DEFAULT_EYE_WIDE_RATIO
            }

            isCalibrated = true

            Log.i("EyeCalibration", String.format(
                "👁️ Eye calibrated: mean=%.4f, std=%.4f, range=[%.4f, %.4f], thresholds: closed=%.4f, open=%.4f, wide=%.4f",
                meanEAR, stdEAR, minEAR, maxEAR, calibratedClosedRatio, calibratedOpenRatio, calibratedWideRatio
            ))
        }

        /**
         * 重置校准状态
         */
        fun reset() {
            sumEAR = 0f
            sumSquaredEAR = 0f
            minEAR = Float.MAX_VALUE
            maxEAR = Float.MIN_VALUE
            frameCount = 0
            startTimestampMs = 0L
            meanEAR = 0f
            stdEAR = 0f
            calibratedClosedRatio = DEFAULT_EYE_CLOSED_RATIO
            calibratedOpenRatio = DEFAULT_EYE_OPEN_RATIO
            calibratedWideRatio = DEFAULT_EYE_WIDE_RATIO
            isCalibrated = false
        }
    }
    
    // 左右眼分别校准
    val imageLeftEyeCalibration = EyeCalibrationData()
    val imageRightEyeCalibration = EyeCalibrationData()

    /**
     * 是否正在校准期
     */
    val isCalibrating: Boolean
        get() = !imageLeftEyeCalibration.isCalibrated || !imageRightEyeCalibration.isCalibrated
    
    /**
     * 重置校准状态
     * 调用此方法后，下一次处理会开始新的校准期
     */
    fun resetCalibration() {
        imageLeftEyeCalibration.reset()
        imageRightEyeCalibration.reset()
        calibrationPitch = 0f
        calibrationYaw = 0f
        calibrationFaceRatio = 0.5f
        Log.i("EyeCalibration", "🔄 Eye calibration reset, will start new calibration")
    }
    
    /**
     * 设置校准时的头部姿态
     * 在校准完成时调用，记录用户校准时的姿态
     * 
     * @param pitch 校准时的俯仰角
     * @param yaw 校准时的偏航角
     */
    fun setCalibrationPose(pitch: Float, yaw: Float) {
        calibrationPitch = pitch
        calibrationYaw = yaw
        calibrationFaceRatio = currentFaceRatio
        Log.i("EyeCalibration", String.format(
            "📐 Calibration pose set: pitch=%.1f°, yaw=%.1f°, faceRatio=%.2f",
            pitch, yaw, calibrationFaceRatio
        ))
    }
    
    /**
     * 获取有效的闭眼阈值（考虑校准、灵敏度和用户偏移）
     * 
     * 灵敏度逻辑：
     * - eyeSensitivity > 1.0: 更灵敏 → closedRatio 变大 → EAR 不需要那么小就能触发闭眼
     * - eyeSensitivity < 1.0: 不灵敏 → closedRatio 变小 → 需要更小的 EAR 才能触发闭眼
     */
    private fun getEffectiveClosedRatio(calibration: EyeCalibrationData): Float {
        val baseRatio = if (calibration.isCalibrated) {
            calibration.calibratedClosedRatio
        } else {
            DEFAULT_EYE_CLOSED_RATIO
        }
        
        val baseOpen = if (calibration.isCalibrated) {
            calibration.calibratedOpenRatio
        } else {
            DEFAULT_EYE_OPEN_RATIO
        }
        
        val baseRange = baseOpen - baseRatio
        val adjustFactor = baseRange * 0.3f
        
        // 灵敏度调整：
        // eyeSensitivity > 1: closedRatio 变大（更容易闭眼）
        // eyeSensitivity < 1: closedRatio 变小（更难闭眼）
        val sensitivityAdjusted = baseRatio + (eyeSensitivity - 1.0f) * adjustFactor
        
        // 应用用户偏移（闭不上时增大此值）
        val withOffset = sensitivityAdjusted + closedThresholdOffset
        
        return withOffset.coerceIn(0.05f, baseOpen - 0.02f)
    }
    
    /**
     * 获取有效的睁眼阈值（考虑校准、灵敏度和用户偏移）
     * 
     * 灵敏度逻辑：
     * - eyeSensitivity > 1.0: 更灵敏 → openRatio 变小 → EAR 不需要那么大就能触发睁眼
     * - eyeSensitivity < 1.0: 不灵敏 → openRatio 变大 → 需要更大的 EAR 才能触发睁眼
     */
    private fun getEffectiveOpenRatio(calibration: EyeCalibrationData): Float {
        val baseClosed = if (calibration.isCalibrated) {
            calibration.calibratedClosedRatio
        } else {
            DEFAULT_EYE_CLOSED_RATIO
        }
        
        val baseRatio = if (calibration.isCalibrated) {
            calibration.calibratedOpenRatio
        } else {
            DEFAULT_EYE_OPEN_RATIO
        }
        
        val baseWide = if (calibration.isCalibrated) {
            calibration.calibratedWideRatio
        } else {
            DEFAULT_EYE_WIDE_RATIO
        }
        
        val baseRange = baseRatio - baseClosed
        val adjustFactor = baseRange * 0.3f
        
        // 灵敏度调整：
        // eyeSensitivity > 1: openRatio 变小（更容易睁眼）
        // eyeSensitivity < 1: openRatio 变大（更难睁眼）
        val sensitivityAdjusted = baseRatio - (eyeSensitivity - 1.0f) * adjustFactor
        
        // 应用用户偏移（睁不开时增大此值，降低 openRatio）
        val withOffset = sensitivityAdjusted - openThresholdOffset
        
        return withOffset.coerceIn(baseClosed + 0.02f, baseWide - 0.02f)
    }


    
    // ============ 用户可调的阈值偏移 ============
    
    /**
     * 闭眼阈值偏移
     * 
     * UI 说明："眼睛闭不上时调整该滑块"
     * 
     * 当用户想闭眼但皮套睁着时，增大此值：
     * - 0.0 = 默认（使用校准后的阈值）
     * - > 0 = 提高 closedRatio，使闭眼更容易触发
     * - < 0 = 降低 closedRatio，使闭眼更难触发
     * 
     * 推荐范围: -0.05 ~ 0.05
     * UI 滑块范围建议: 0% ~ 100% 映射到 0.0 ~ 0.05
     */
    var closedThresholdOffset: Float = 0f
        set(value) {
            field = value.coerceIn(-0.08f, 0.08f)
        }
    
    /**
     * 睁眼阈值偏移
     * 
     * UI 说明："眼睛睁不开时调整该滑块"
     * 
     * 当用户睁着眼但皮套闭着时，增大此值：
     * - 0.0 = 默认（使用校准后的阈值）
     * - > 0 = 降低 openRatio，使睁眼更容易触发
     * - < 0 = 提高 openRatio，使睁眼更难触发
     * 
     * 推荐范围: -0.05 ~ 0.05
     * UI 滑块范围建议: 0% ~ 100% 映射到 0.0 ~ 0.05
     */
    var openThresholdOffset: Float = 0f
        set(value) {
            field = value.coerceIn(-0.08f, 0.08f)
        }

    /**
     * 获取当前有效的 EAR 阈值范围（用于调试显示）
     * 使用左眼的校准数据
     * @return Pair(closedRatio, openRatio)
     */
    fun getEffectiveThresholds(): Pair<Float, Float> {
        return Pair(
            getEffectiveClosedRatio(imageLeftEyeCalibration),
            getEffectiveOpenRatio(imageLeftEyeCalibration)
        )
    }
    
    // ============ 头部姿态补偿参数 ============
    // 这些系数控制补偿强度，可以根据实际效果调整
    
    /**
     * ==================== 动态范围补偿模型 ====================
     * 
     * 核心思想：
     * 头部俯仰角会导致 EAR 的动态范围收缩，而不是简单的偏移。
     * 
     * 正对相机时:
     * EAR: [0.10 闭眼] ←──────────────────────────→ [0.18 睁眼]
     *                     正常范围 = 0.08
     * 
     * 低头/抬头时:
     * EAR: [0.11 闭眼] ←─────────────→ [0.15 睁眼]
     *              收缩范围 = 0.04
     * 
     * 补偿策略：将收缩后的范围线性映射回正常范围
     * 
     * 公式：
     * compensated = (raw - shrunk_min) / (shrunk_max - shrunk_min) 
     *             × (normal_max - normal_min) + normal_min
     * 
     * 这样：
     * - 睁眼时：收缩后的上限 → 映射到正常上限
     * - 闭眼时：收缩后的下限 → 映射到正常下限
     * - 中间值：线性插值
     */
    
    /**
     * 范围收缩系数（每度 pitch 导致的范围收缩比例）
     * 
     * 例如：shrinkFactorPerDegree = 0.015 表示每度 pitch 会导致范围收缩 1.5%
     * 当 pitch = 30° 时，范围收缩到原来的 1 - 30 * 0.015 = 55%
     */
    var rangeShrinkFactorPerDegree: Float = 0.012f
    
    /**
     * 低头时的额外收缩倍数（因为上眼皮遮挡效应）
     * 1.0 = 与抬头相同
     * 1.3 = 低头时收缩更严重
     */
    var pitchDownShrinkMultiplier: Float = 1.2f
    
    /**
     * 最小收缩比例（防止范围过度收缩导致数值不稳定）
     * 例如：0.3 表示范围最多收缩到原来的 30%
     */
    var minShrinkRatio: Float = 0.4f
    
    /**
     * 计算在给定 pitch 角度下的范围收缩比例
     * 
     * @param pitch 俯仰角（度）
     * @return 收缩比例 [minShrinkRatio, 1.0]，1.0 表示无收缩
     */
    fun calculateShrinkRatio(pitch: Float): Float {
        val absPitch = abs(pitch)
        if (absPitch <= pitchCompensationThreshold) {
            return 1.0f  // 在阈值内，无收缩
        }
        
        val effectivePitch = absPitch - pitchCompensationThreshold
        var shrinkAmount = effectivePitch * rangeShrinkFactorPerDegree
        
        // 低头时额外收缩
        if (pitch < 0) {
            shrinkAmount *= pitchDownShrinkMultiplier
        }
        
        val shrinkRatio = (1.0f - shrinkAmount).coerceIn(minShrinkRatio, 1.0f)
        return shrinkRatio
    }
    
    /**
     * 应用动态范围补偿
     * 
     * 注意：这个补偿逻辑假设头部俯仰会导致 EAR 整体偏小。
     * 但实际上，当真的闭眼时，EAR 应该还是很小，不需要被"拉大"。
     * 
     * 修正后的策略：
     * - 只补偿"本来应该是睁眼但因为头部姿态而变小"的情况
     * - 不补偿真正的闭眼（EAR 很小的情况）
     * 
     * @param ear 原始 EAR 值
     * @param pitch 俯仰角（度）
     * @param calibration 使用的校准数据
     * @return 补偿后的 EAR 值
     */
    fun compensateEarForPitch(ear: Float, pitch: Float, calibration: EyeCalibrationData): Float {
        if (abs(pitch) <= pitchCompensationThreshold) {
            return ear  // 在阈值内，不补偿
        }
        
        val normalClosed = getEffectiveClosedRatio(calibration)
        val normalOpen = getEffectiveOpenRatio(calibration)
        
        // 如果 EAR 已经很小（接近或低于闭眼阈值），说明是真的在闭眼
        // 不应该补偿，否则会导致闭眼变困难
        if (ear <= normalClosed * 1.2f) {
            return ear  // 真正闭眼，不补偿
        }
        
        // 计算收缩比例
        val shrinkRatio = calculateShrinkRatio(pitch)
        
        // 只对"中间区域"的 EAR 进行补偿
        // 思路：头部俯仰导致睁眼时的 EAR 变小，但闭眼时的 EAR 基本不变
        // 所以我们只需要把"看起来像半闭眼但实际是睁眼"的情况补偿回来
        
        // 假设闭眼时的 EAR 基本不受头部姿态影响（因为眼睛真的闭上了）
        // 只有睁眼时的上限被压缩了
        val shrunkOpen = normalClosed + (normalOpen - normalClosed) * shrinkRatio
        
        // 只对 [normalClosed, shrunkOpen] 区间内的值进行映射
        if (ear >= shrunkOpen) {
            // 已经在"睁眼"区域，映射到正常的睁眼上限
            return normalOpen
        }
        
        if (ear <= normalClosed) {
            // 在闭眼区域，不补偿
            return ear
        }
        
        // 在中间区域，线性映射
        // [normalClosed, shrunkOpen] -> [normalClosed, normalOpen]
        val shrunkRange = shrunkOpen - normalClosed
        val normalRange = normalOpen - normalClosed
        
        if (shrunkRange < 0.01f) {
            return ear
        }
        
        val normalizedPosition = (ear - normalClosed) / shrunkRange
        val compensatedEar = normalizedPosition * normalRange + normalClosed
        
        return compensatedEar
    }
    
    // ============ 传统 Offset 补偿参数（向后兼容）============
    
    /**
     * Pitch 补偿开始角度
     * 只有当 |pitch| > 此角度时才开始补偿
     */
    var pitchCompensationThreshold: Float = 5f
    
    /**
     * Yaw 补偿系数 (偏航角)
     * 
     * 镜像关系 (前置摄像头):
     * - 真人向右转 (yaw > 0): 真人右眼被遮挡 → 图像左眼 → eyeBlinkLeft 补偿
     * - 真人向左转 (yaw < 0): 真人左眼被遮挡 → 图像右眼 → eyeBlinkRight 补偿
     * 
     * 补偿公式: 被遮挡眼的 blinkCompensation = -|yaw| * YAW_COMPENSATION_FACTOR / 90
     * 范围: 0.0 = 不补偿, 0.2 = 轻微补偿, 0.4 = 中等补偿
     */
    var yawCompensationFactor: Float = 0.4f
    
    /**
     * Yaw 补偿开始角度
     * 只有当 |yaw| > 此角度时才开始补偿
     */
    var yawCompensationThreshold: Float = 1f
    
    /**
     * 近侧眼补偿倍数（相对于远侧眼）
     * 
     * 当转头时，近侧眼因为透视变形，EAR 下降比远侧眼更严重。
     * 透视变形的强度取决于：
     * - 人脸到相机的距离（越近越强）
     * - 相机 FOV（广角越强）
     * - 人脸在画面中的占比（占比越大表示越近）
     * 
     * 经验值：
     * - 人脸占画面 70%+（近距离）：5.0 ~ 6.0
     * - 人脸占画面 50%（中距离）：3.0 ~ 4.0
     * - 人脸占画面 30%（远距离）：1.5 ~ 2.0
     * 
     * 如果 enableDynamicCompensation = true，此值会被动态计算覆盖
     */
    var nearSideCompensationMultiplier: Float = 5.0f
        set(value) {
            field = value.coerceIn(0.5f, 10.0f)
        }
    
    // ============ 动态补偿系数（基于人脸大小和 FOV）============
    
    /**
     * ==================== 透视补偿物理模型 ====================
     * 
     * 透视变形公式：
     * 
     * 当人脸转动 yaw 角度时，近侧眼和远侧眼到相机的距离差：
     *   Δz = faceWidth × sin(yaw)
     * 
     * 透视导致的缩放差：
     *   scaleDiff ≈ Δz / distance ≈ (faceWidth / distance) × sin(yaw)
     * 
     * 由于 faceRatio ≈ faceWidth / distance（人脸在画面中的占比），所以：
     *   scaleDiff ≈ faceRatio × sin(yaw)
     * 
     * 同时，FOV 越大，透视效果越强：
     *   perspectiveStrength ∝ tan(FOV/2)
     * 
     * 综合补偿模型：
     *   compensationMultiplier = baseMultiplier × (1 + perspectiveStrength × distanceFactor)
     * 
     * 其中：
     *   - perspectiveStrength = tan(FOV/2) / tan(30°)  // 归一化到 60° FOV
     *   - distanceFactor = (faceRatio - referenceRatio) / referenceRatio
     */
    
    /**
     * 是否启用基于人脸大小的动态补偿系数
     */
    var enableDynamicCompensation: Boolean = true
    
    /**
     * 相机垂直 FOV（度）
     * 典型手机前置摄像头 FOV 在 60-80° 之间
     * 如果不知道具体值，可以使用默认值 63°
     */
    var cameraFovDegrees: Float = 63f
        set(value) {
            field = value.coerceIn(30f, 120f)
            updatePerspectiveStrength()
        }
    
    /**
     * 透视强度（内部计算值）
     * = tan(FOV/2) / tan(30°)
     * 
     * 标准化到 60° FOV = 1.0
     */
    private var perspectiveStrength: Float = 1.0f
    
    /**
     * 更新透视强度
     */
    private fun updatePerspectiveStrength() {
        val halfFovRad = Math.toRadians(cameraFovDegrees / 2.0).toFloat()
        val referenceFovRad = Math.toRadians(30.0).toFloat()  // 60° FOV 的一半
        perspectiveStrength = tan(halfFovRad) / tan(referenceFovRad)
    }
    
    /**
     * 参考人脸占比（基准点）
     * 当人脸占比等于此值时，使用 baseMultiplier
     */
    var referenceRatioRatio: Float = 0.4f
    
    /**
     * 基础补偿倍数
     * 在参考距离、参考 FOV 下的补偿系数
     */
    var baseCompensationMultiplier: Float = 3.0f
    
    /**
     * 距离敏感度
     * 控制距离变化对补偿系数的影响强度
     * 
     * 公式：multiplier = base × (1 + distanceSensitivity × perspectiveStrength × distanceFactor)
     */
    var distanceSensitivity: Float = 2.5f
    
    /**
     * 当前估算的人脸距离比例
     * 由 updateFaceDistanceEstimate() 更新
     */
    private var currentFaceRatio: Float = 0.5f
    
    /**
     * 使用归一化的人脸框坐标更新距离估计
     * 
     * @param normalizedWidth 人脸框宽度（归一化 0-1）
     * @param normalizedHeight 人脸框高度（归一化 0-1）
     */
    fun updateFaceDistanceEstimateNormalized(
        normalizedWidth: Float,
        normalizedHeight: Float
    ) {
        // 检查输入有效性
        if (normalizedWidth <= 0 || normalizedHeight <= 0 || 
            normalizedWidth > 1.5f || normalizedHeight > 1.5f) {
            return  // 无效输入，跳过
        }
        
        // 归一化坐标下，直接使用宽高乘积的平方根作为 faceRatio
        val faceRatio = sqrt(normalizedWidth * normalizedHeight)
        
        // 平滑更新（避免抖动）
        currentFaceRatio = currentFaceRatio * 0.9f + faceRatio * 0.1f
        
        if (enableDynamicCompensation) {
            // 基于物理模型计算补偿系数
            // distanceFactor: 人脸比参考距离近多少（正值=更近）
            val distanceFactor = (currentFaceRatio - referenceRatioRatio) / referenceRatioRatio
            
            // 综合公式：base × (1 + sensitivity × perspective × distance)
            val dynamicMultiplier = baseCompensationMultiplier * 
                (1f + distanceSensitivity * perspectiveStrength * distanceFactor.coerceIn(-0.5f, 1.5f))
            
            val prevMultiplier = nearSideCompensationMultiplier
            nearSideCompensationMultiplier = dynamicMultiplier.coerceIn(0.5f, 15.0f)
            
            // 当补偿系数变化较大时打印日志
            if (abs(nearSideCompensationMultiplier - prevMultiplier) > 0.3f) {
                Log.d("EyeExtractor", String.format(
                    "📏 faceRatio=%.2f, FOV=%.0f°, perspective=%.2f -> multiplier=%.1f",
                    currentFaceRatio, cameraFovDegrees, perspectiveStrength, nearSideCompensationMultiplier
                ))
            }
        }
    }
    
    /**
     * 设置相机 FOV（从相机参数获取）
     */
    fun setCameraFov(fovDegrees: Float) {
        cameraFovDegrees = fovDegrees
        Log.i("EyeExtractor", "📷 Camera FOV set to: $fovDegrees°, perspectiveStrength=$perspectiveStrength")
    }
    
    // 初始化时计算透视强度
    init {
        updatePerspectiveStrength()
    }

    /**
     * 是否启用头部姿态补偿
     */
    var enableHeadPoseCompensation: Boolean = true

    /**
     * 使用标准 Eye Aspect Ratio (EAR) 算法
     * EAR = (vertical_distance) / (horizontal_distance)
     * 
     * 参考论文: "Real-Time Eye Blink Detection using Facial Landmarks"
     */
    fun computeEyeAspectRatio(eye: EyePoints): Float {
        // 计算垂直距离 (上下眼睑)
        val verticalDistances = mutableListOf<Float>()
        
        // 对于每个上眼睑点，找到最近的下眼睑点
        for (upperPt in eye.upper) {
            val minDist = eye.lower.minOfOrNull { lowerPt ->
                upperPt.distance2DTo(lowerPt)
            } ?: 0f
            verticalDistances.add(minDist)
        }
        
        val avgVertical = if (verticalDistances.isNotEmpty()) {
            verticalDistances.average().toFloat()
        } else {
            0f
        }
        
        // 计算水平距离 (眼角之间)
        val horizontal = eye.innerCorner.distance2DTo(eye.outerCorner)
        
        // EAR = vertical / horizontal
        return if (horizontal > 1e-6f) {
            avgVertical / horizontal
        } else {
            0f
        }
    }
    
    /**
     * 从 EAR 直接映射到 eyeBlink [0,1]
     * 0 = 完全睁开
     * 1 = 完全闭合
     * @param ear EAR 值
     * @param calibration 使用的校准数据
     */
    fun earToBlinkValue(ear: Float, calibration: EyeCalibrationData): Float {
        val effectiveClosedRatio = getEffectiveClosedRatio(calibration)
        val effectiveOpenRatio = getEffectiveOpenRatio(calibration)
        
        return when {
            ear <= effectiveClosedRatio -> 1.0f  // 完全闭合
            ear >= effectiveOpenRatio -> 0.0f    // 完全睁开
            else -> {
                // 线性插值
                val range = effectiveOpenRatio - effectiveClosedRatio
                val offset = effectiveOpenRatio - ear
                (offset / range).coerceIn(0f, 1f)
            }
        }
    }
    
    /**
     * 检测眼睛睁大
     * 0 = 正常
     * 1 = 最大睁大
     * 
     * @param ear EAR 值
     * @param calibration 使用的校准数据
     */
    fun detectEyeWide(ear: Float, calibration: EyeCalibrationData): Float {
        val openRatio = getEffectiveOpenRatio(calibration)
        val wideRatio = if (calibration.isCalibrated) {
            calibration.calibratedWideRatio
        } else {
            DEFAULT_EYE_WIDE_RATIO
        }
        
        return if (ear > openRatio) {
            val wideAmount = (ear - openRatio) / (wideRatio - openRatio)
            wideAmount.coerceIn(0f, 1f)
        } else {
            0f
        }
    }
    
    /**
     * 完整的眼部 BlendShape 提取
     * 
     * Pipeline 设计：
     * 1. 计算原始 EAR
     * 2. 头部姿态补偿 EAR（恢复被压缩的动态范围）← 在转换之前！
     * 3. 校准期收集统计
     * 4. 使用校准阈值转换为 BlendShape
     * 
     * @param imageLeftEye 左眼关键点
     * @param imageRightEye 右眼关键点
     * @param blendShapes 输出的 BlendShape 对象
     * @param headPose 头部姿态角 (pitch, yaw, roll)，可为 null
     */
    fun extractEyeBlendShapes(
        imageLeftEye: EyePoints,
        imageRightEye: EyePoints,
        blendShapes: ARKitBlendShapes,
        headPose: HeadPose?
    ) {
        // ========== Step 1: 计算原始 EAR ==========
        val imageLeftEARRaw = computeEyeAspectRatio(imageLeftEye)
        val imageRightEARRaw = computeEyeAspectRatio(imageRightEye)
        
        // ========== Step 2: 头部姿态补偿 EAR（在转换之前！）==========
        // 核心思想：头部俯仰/偏航会压缩 EAR 的动态范围
        // 补偿应该在 EAR 层面进行，恢复被压缩的范围
        var imageLeftEAR = imageLeftEARRaw
        var imageRightEAR = imageRightEARRaw
        
        // 当 pitch 接近 0 时，完全跳过补偿
        val pitchNearZero = headPose == null || abs(headPose.pitch) <= pitchCompensationThreshold
        val yawNearZero = headPose == null || abs(headPose.yaw) <= yawCompensationThreshold
        
        if (enableHeadPoseCompensation && headPose != null && !pitchNearZero) {
            // 对 EAR 进行动态范围补偿
            imageLeftEAR = compensateEarForPitch(imageLeftEARRaw, headPose.pitch, imageLeftEyeCalibration)
            imageRightEAR = compensateEarForPitch(imageRightEARRaw, headPose.pitch, imageRightEyeCalibration)
        }
        
        if (enableHeadPoseCompensation && headPose != null && !yawNearZero) {
            // ==================== 相对补偿策略 ====================
            // 补偿的是"相对于校准时刻的透视变化量"
            // 而不是"相对于正脸的透视量"
            //
            // 公式：compensation = f(currentYaw) - f(calibrationYaw)
            //
            // 这样当用户保持校准时的姿态时，补偿量 = 0
            
            // 计算当前姿态的补偿量
            val currentAbsYaw = abs(headPose.yaw)
            val currentYawFactor = maxOf(0f, (currentAbsYaw - yawCompensationThreshold) / 90f)
            
            // 计算校准时姿态的补偿量（基线）
            val calibrationAbsYaw = abs(calibrationYaw)
            val calibrationYawFactor = maxOf(0f, (calibrationAbsYaw - yawCompensationThreshold) / 90f)
            
            // 相对变化量
            val deltaYawFactor = currentYawFactor - calibrationYawFactor
            
            val leftClosedRatio = getEffectiveClosedRatio(imageLeftEyeCalibration)
            val leftOpenRatio = getEffectiveOpenRatio(imageLeftEyeCalibration)
            val rightClosedRatio = getEffectiveClosedRatio(imageRightEyeCalibration)
            val rightOpenRatio = getEffectiveOpenRatio(imageRightEyeCalibration)
            
            val leftEarRange = leftOpenRatio - leftClosedRatio
            val rightEarRange = rightOpenRatio - rightClosedRatio
            
            // ==================== 自适应补偿衰减 ====================
            // 核心思想：
            // - 睁眼时（EAR 高）：透视导致 EAR 偏小，需要完全补偿
            // - 闭眼时（EAR 低）：眼睛真的闭上了，不需要补偿
            //
            // 衰减公式：
            // decay = (currentEAR - closedRatio) / (openRatio - closedRatio)
            // decay = 0 表示完全闭眼，不补偿
            // decay = 1 表示完全睁眼，完全补偿
            
            val leftDecay = ((imageLeftEARRaw - leftClosedRatio) / leftEarRange).coerceIn(0f, 1f)
            val rightDecay = ((imageRightEARRaw - rightClosedRatio) / rightEarRange).coerceIn(0f, 1f)
            
            // 只有当 deltaYawFactor > 0（比校准时转头更多）时才需要补偿
            // 当 deltaYawFactor < 0（转头比校准时少）时，不做额外补偿
            if (deltaYawFactor > 0) {
                // 远侧眼补偿：主要是遮挡导致
                // 近侧眼补偿：透视变形导致，需要更大的补偿系数
                val farSideCompensation = deltaYawFactor * yawCompensationFactor
                val nearSideCompensation = deltaYawFactor * yawCompensationFactor * nearSideCompensationMultiplier
                
                // 判断当前转头方向，并应用衰减
                if (headPose.yaw > 0) {
                    // 真人向左转：左眼是远侧（被遮挡），右眼是近侧（透视变形）
                    imageLeftEAR += farSideCompensation * leftEarRange * leftDecay
                    imageRightEAR += nearSideCompensation * rightEarRange * rightDecay
                } else {
                    // 真人向右转：右眼是远侧（被遮挡），左眼是近侧（透视变形）
                    imageRightEAR += farSideCompensation * rightEarRange * rightDecay
                    imageLeftEAR += nearSideCompensation * leftEarRange * leftDecay
                }
            }
            
            // 处理转头方向改变的情况
            // 如果校准时向左转，现在向右转，需要双向补偿
            val calibrationDirection = if (calibrationYaw >= 0) 1 else -1
            val currentDirection = if (headPose.yaw >= 0) 1 else -1
            
            if (calibrationDirection != currentDirection && calibrationAbsYaw > yawCompensationThreshold) {
                // 方向改变，需要额外补偿校准时的偏移
                val oppositeCompensation = calibrationYawFactor * yawCompensationFactor
                val oppositeNearCompensation = calibrationYawFactor * yawCompensationFactor * nearSideCompensationMultiplier
                
                if (currentDirection > 0) {
                    // 现在向左转，校准时是向右转
                    // 校准时右眼是远侧，左眼是近侧
                    // 现在需要反向补偿（同样应用衰减）
                    imageRightEAR += oppositeCompensation * rightEarRange * rightDecay
                    imageLeftEAR += oppositeNearCompensation * leftEarRange * leftDecay
                } else {
                    // 现在向右转，校准时是向左转
                    imageLeftEAR += oppositeCompensation * leftEarRange * leftDecay
                    imageRightEAR += oppositeNearCompensation * rightEarRange * rightDecay
                }
            }
        }
        
        // ========== Step 3: 自适应校准期 ==========
        // 使用原始 EAR 进行校准（不含补偿）
        // 并在校准完成时记录当时的头部姿态
        if (!imageLeftEyeCalibration.isCalibrated) {
            imageLeftEyeCalibration.addFrame(imageLeftEARRaw)
            if (imageLeftEyeCalibration.isWindowReady()) {
                imageLeftEyeCalibration.finishCalibration()
                // 记录校准时的姿态
                if (headPose != null) {
                    calibrationPitch = headPose.pitch
                    calibrationYaw = headPose.yaw
                    calibrationFaceRatio = currentFaceRatio
                    Log.i("EyeCalibration", String.format(
                        "📐 Left eye calibrated at pose: pitch=%.1f°, yaw=%.1f°, faceRatio=%.2f",
                        calibrationPitch, calibrationYaw, calibrationFaceRatio
                    ))
                }
            }
        }
        if (!imageRightEyeCalibration.isCalibrated) {
            imageRightEyeCalibration.addFrame(imageRightEARRaw)
            if (imageRightEyeCalibration.isWindowReady()) {
                imageRightEyeCalibration.finishCalibration()
            }
        }

        // ========== Step 4: EAR 转换为 BlendShape ==========
        // 使用补偿后的 EAR 和校准后的阈值
        val imageLeftBlink = earToBlinkValue(imageLeftEAR, imageLeftEyeCalibration)
        val imageRightBlink = earToBlinkValue(imageRightEAR, imageRightEyeCalibration)
        val imageLeftEyeWide = detectEyeWide(imageLeftEAR, imageLeftEyeCalibration)
        val imageRightEyeWide = detectEyeWide(imageRightEAR, imageRightEyeCalibration)
        
        // 赋值到 BlendShape（不再需要额外的 offset 补偿）
        blendShapes.eyeBlinkLeft = imageLeftBlink
        blendShapes.eyeBlinkRight = imageRightBlink
        blendShapes.eyeWideLeft = imageLeftEyeWide
        blendShapes.eyeWideRight = imageRightEyeWide
    }
}
