package com.live2d.facecapture

import android.os.SystemClock
import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.CALIBRATION_DURATION_MS
import com.example.commondata.FaceKeyPoints
import com.example.commondata.HeadPose
import com.example.commondata.MIN_CALIBRATION_FRAMES
import com.example.commondata.Point3D
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Author: Voine
 * Date: 2025/12/30
 * Description: 嘴部 BlendShape 提取器，支持头部姿态补偿
 * 
 * ==================== 俯仰角补偿原理 ====================
 * 
 * 问题：
 * 当用户低头/抬头时，嘴巴的可见高度会因透视变化而压缩：
 * - 低头时 (pitch > 0)：嘴巴被下巴遮挡，MAR 变小
 * - 抬头时 (pitch < 0)：嘴巴被鼻子遮挡，MAR 也可能变小
 * 
 * 这会导致用户张嘴时 jawOpen 不够大，或者需要张得更大才能达到同样效果。
 * 
 * 补偿策略：
 * 与眼睛补偿类似，计算"相对于校准时刻的透视变化量"
 * compensated = raw + [f(currentPitch) - f(calibrationPitch)]
 */
class SimpleMouthExtractor {

    companion object {
        // ============ 嘴巴检测默认阈值 ============

        // 嘴巴开合阈值 (Mouth Aspect Ratio - MAR)
        private const val DEFAULT_MOUTH_CLOSED_RATIO = 0.10f  // MAR < 此值认为闭嘴
        private const val DEFAULT_MOUTH_OPEN_RATIO = 0.45f    // MAR > 此值认为张大嘴
    }
    
    // ============ 自适应校准系统 ============
    
    /**
     * 嘴部校准数据类
     */
    data class MouthCalibrationData(
        // 累积统计
        var sumMAR: Float = 0f,
        var sumSquaredMAR: Float = 0f,
        var minMAR: Float = Float.MAX_VALUE,
        var maxMAR: Float = Float.MIN_VALUE,
        var frameCount: Int = 0,
        // 校准窗口起始时间（uptimeMillis），用于按时长触发结束
        var startTimestampMs: Long = 0L,

        // 校准结果
        var meanMAR: Float = 0f,
        var stdMAR: Float = 0f,
        var calibratedClosedRatio: Float = DEFAULT_MOUTH_CLOSED_RATIO,
        var calibratedOpenRatio: Float = DEFAULT_MOUTH_OPEN_RATIO,
        
        var isCalibrated: Boolean = false
    ) {
        /**
         * 添加一帧 MAR 数据
         */
        fun addFrame(mar: Float) {
            if (frameCount == 0) {
                startTimestampMs = SystemClock.uptimeMillis()
            }
            sumMAR += mar
            sumSquaredMAR += mar * mar
            minMAR = min(minMAR, mar)
            maxMAR = max(maxMAR, mar)
            frameCount++
        }

        /**
         * 校准窗口是否已经收集够样本（时长 + 最小帧数 双门限）。
         */
        fun isWindowReady(): Boolean {
            if (frameCount < MIN_CALIBRATION_FRAMES) return false
            val elapsed = SystemClock.uptimeMillis() - startTimestampMs
            return elapsed >= CALIBRATION_DURATION_MS
        }
        
        /**
         * 完成校准，计算阈值
         * 假设校准期间用户闭嘴，MAR 应该较小
         */
        fun finishCalibration() {
            if (frameCount < 10) {
                Log.w("MouthCalibration", "Too few frames for calibration: $frameCount")
                return
            }
            
            // 计算统计值
            meanMAR = sumMAR / frameCount
            val variance = (sumSquaredMAR / frameCount) - (meanMAR * meanMAR)
            stdMAR = sqrt(max(0f, variance))
            
            // 校准时用户应该闭嘴，所以 meanMAR 就是闭嘴时的 MAR
            // closedRatio = meanMAR + 一些余量（允许小幅抖动）
            calibratedClosedRatio = (meanMAR + 2f * stdMAR).coerceIn(0.05f, 0.25f)
            
            // openRatio 基于 closedRatio 推断
            // 假设张大嘴时 MAR 是闭嘴时的 3-4 倍
            calibratedOpenRatio = (calibratedClosedRatio * 3.5f).coerceIn(0.25f, 0.70f)
            
            isCalibrated = true
            
            Log.i("MouthCalibration", String.format(
                "👄 Mouth calibrated: mean=%.4f, std=%.4f, thresholds: closed=%.4f, open=%.4f",
                meanMAR, stdMAR, calibratedClosedRatio, calibratedOpenRatio
            ))
        }
        
        /**
         * 重置校准状态
         */
        fun reset() {
            sumMAR = 0f
            sumSquaredMAR = 0f
            minMAR = Float.MAX_VALUE
            maxMAR = Float.MIN_VALUE
            frameCount = 0
            startTimestampMs = 0L
            meanMAR = 0f
            stdMAR = 0f
            calibratedClosedRatio = DEFAULT_MOUTH_CLOSED_RATIO
            calibratedOpenRatio = DEFAULT_MOUTH_OPEN_RATIO
            isCalibrated = false
        }
    }
    
    // 校准数据
    val mouthCalibration = MouthCalibrationData()

    /**
     * 是否正在校准期
     */
    val isCalibrating: Boolean
        get() = !mouthCalibration.isCalibrated
    
    /**
     * 校准时的头部姿态角
     */
    private var calibrationPitch: Float = 0f
    private var calibrationYaw: Float = 0f
    private var calibrationRoll: Float = 0f
    
    // ============ 俯仰角补偿参数 ============
    
    /**
     * 是否启用俯仰角补偿
     */
    var enablePitchCompensation: Boolean = true
    
    /**
     * 俯仰角补偿开始阈值（度）
     * 只有当 |pitch - calibrationPitch| > 此值时才补偿
     */
    var pitchCompensationThreshold: Float = 5f
    
    /**
     * 俯仰角补偿系数
     * 每度俯仰变化对 MAR 的影响
     * 
     * 物理意义：当头部俯仰时，嘴巴的可见高度按 cos(pitch) 压缩
     * 但实际效果比纯几何更复杂（有遮挡等因素），所以使用经验系数
     */
    var pitchCompensationFactor: Float = 0.008f
    
    /**
     * 低头时的额外补偿倍数（因为下巴遮挡更严重）
     */
    var pitchDownMultiplier: Float = 1.3f

    /**
     * 设置校准时的头部姿态
     * @param pitch 校准时的俯仰角（度）
     * @param yaw 校准时的偏航角（度）
     * @param roll 校准时的翻滚角（度）
     */
    fun setCalibrationPose(pitch: Float, yaw: Float, roll: Float) {
        calibrationPitch = pitch
        calibrationYaw = yaw
        calibrationRoll = roll
        Log.i("SimpleMouthExtractor", String.format(
            "📐 Mouth calibration pose set: pitch=%.1f°, yaw=%.1f°, roll=%.1f°",
            pitch, yaw, roll
        ))
    }

    /**
     * 计算俯仰角补偿量
     * 
     * @param currentPitch 当前俯仰角
     * @param rawMAR 当前原始 MAR 值，用于智能判断是否需要补偿
     * @return MAR 补偿量（正值表示需要增大 MAR）
     * 
     * 修复说明：
     * 原逻辑问题：低头时无条件增大 MAR，导致闭嘴时也误报张嘴
     * 新逻辑：只有当 rawMAR 已经超过闭嘴阈值时才进行补偿，
     *        避免闭嘴状态被错误放大
     */
    private fun calculatePitchCompensation(currentPitch: Float, rawMAR: Float): Float {
        // 计算相对于校准时刻的俯仰变化
        val deltaPitch = currentPitch - calibrationPitch
        
        if (abs(deltaPitch) <= pitchCompensationThreshold) {
            return 0f
        }
        
        // 获取当前的闭嘴阈值
        val closedThreshold = if (mouthCalibration.isCalibrated) {
            mouthCalibration.calibratedClosedRatio
        } else {
            DEFAULT_MOUTH_CLOSED_RATIO
        }
        
        // 关键修复：只有当嘴巴已经有一定程度张开时才补偿
        // 如果 rawMAR 小于闭嘴阈值的 1.5 倍，认为用户是闭嘴状态，不需要补偿
        if (rawMAR < closedThreshold * 1.5f) {
            return 0f
        }
        
        val effectiveDelta = if (deltaPitch > 0) {
            // 低头，需要更多补偿
            (deltaPitch - pitchCompensationThreshold) * pitchDownMultiplier
        } else {
            // 抬头
            -(abs(deltaPitch) - pitchCompensationThreshold)
        }
        
        // 补偿量与角度变化成正比
        return effectiveDelta * pitchCompensationFactor
    }

    fun extractMouthBlendShapes(
        keyPoints: FaceKeyPoints,
        blendShapes: ARKitBlendShapes,
        headPose: HeadPose? = null,
        landmarks: List<Point3D>? = null
    ) {
        val mouthInner = keyPoints.mouthInner
        
        // 1. 计算原始 MAR
        val rawMAR = computeMouthAspectRatio(mouthInner)
        
        // 2. 俯仰角补偿
        var compensatedMAR = rawMAR
        if (enablePitchCompensation && headPose != null && mouthCalibration.isCalibrated) {
            val compensation = calculatePitchCompensation(headPose.pitch, rawMAR)
            compensatedMAR = (rawMAR + compensation).coerceAtLeast(0f)
        }
        
        // 3. 校准期收集数据（使用原始 MAR）
        if (!mouthCalibration.isCalibrated) {
            mouthCalibration.addFrame(rawMAR)
            if (mouthCalibration.isWindowReady()) {
                mouthCalibration.finishCalibration()
            }
        }
        
        // 4. 转换为 jawOpen
        blendShapes.jawOpen = marToJawOpen(compensatedMAR)
    }

    /**
     * 计算 Mouth Aspect Ratio (MAR)
     * MAR = (mouth_height) / (mouth_width)
     */
    fun computeMouthAspectRatio(mouthInner: List<Point3D>): Float {
        // 上唇中心点和下唇中心点
        val upperLipTop = mouthInner[5]     // 上唇内边距中心
        val lowerLipBottom = mouthInner[15] // 下唇内边距中心

        // 左右嘴角
        val leftCorner = mouthInner[0]
        val rightCorner = mouthInner[10]

        // 计算垂直距离和水平距离
        val mouthHeight = upperLipTop.distance2DTo(lowerLipBottom)
        val mouthWidth = leftCorner.distance2DTo(rightCorner)

        // MAR = height / width
        return if (mouthWidth > 1e-6f) {
            mouthHeight / mouthWidth
        } else {
            0f
        }
    }

    /**
     * 从 MAR 映射到 jawOpen [0,1]
     * 0 = 完全闭合
     * 1 = 完全张开
     */
    fun marToJawOpen(mar: Float): Float {
        val closedRatio = if (mouthCalibration.isCalibrated) {
            mouthCalibration.calibratedClosedRatio
        } else {
            DEFAULT_MOUTH_CLOSED_RATIO
        }
        
        val openRatio = if (mouthCalibration.isCalibrated) {
            mouthCalibration.calibratedOpenRatio
        } else {
            DEFAULT_MOUTH_OPEN_RATIO
        }
        
        return when {
            mar <= closedRatio -> 0.0f
            mar >= openRatio -> 1.0f
            else -> {
                // 线性插值
                val range = openRatio - closedRatio
                val offset = mar - closedRatio
                (offset / range).coerceIn(0f, 1f)
            }
        }
    }

    /**
     * 重置校准
     */
    fun resetCalibration() {
        mouthCalibration.reset()
        calibrationPitch = 0f
        calibrationYaw = 0f
        calibrationRoll = 0f
        Log.i("SimpleMouthExtractor", "🔄 Mouth calibration reset")
    }

    fun reset() {
        resetCalibration()
    }
}