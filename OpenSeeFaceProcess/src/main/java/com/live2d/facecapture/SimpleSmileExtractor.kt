package com.live2d.facecapture

import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.FRAME_COUNT_CALIBRATE
import com.example.commondata.FaceKeyPoints
import com.example.commondata.HeadPose
import com.example.commondata.Point3D
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Author: Voine
 * Date: 2025/12/30
 * 
 * 微笑/撇嘴 BlendShape 提取器
 * 
 * ==================== 设计原理 ====================
 * 
 * 微笑检测基于嘴角相对于嘴唇中心的垂直位置：
 * - 微笑时：嘴角上扬，嘴角 Y 坐标 < 嘴唇中心 Y 坐标
 * - 撇嘴时：嘴角下垂，嘴角 Y 坐标 > 嘴唇中心 Y 坐标
 * 
 * 为了消除头部姿态的影响，我们使用相对量：
 * smileFeature = (mouthCenter.y - cornerAvg.y) / faceHeight
 * 
 * 正值 = 微笑（嘴角高于中心）
 * 负值 = 撇嘴（嘴角低于中心）
 * 
 * ==================== 俯仰角补偿 ====================
 * 
 * 当用户低头时，嘴角相对于嘴唇中心的投影位置会发生变化：
 * - 低头 (pitch > 0)：嘴角看起来更低（相对于嘴唇中心）
 * - 抬头 (pitch < 0)：嘴角看起来更高
 * 
 * 补偿策略：计算相对于校准时刻的变化量
 * 
 * ==================== Live2D 输出 ====================
 * 
 * Live2D 使用 ParamMouthForm 参数：
 * - 范围: [-1, 1] 或 [0, 1]，取决于模型
 * - 正值 = 微笑
 * - 负值 = 撇嘴
 * 
 * 计算方式: mouthForm = (smileLeft + smileRight) / 2 - (frownLeft + frownRight) / 2
 */
class SimpleSmileExtractor {

    companion object {
        private const val TAG = "SimpleSmileExtractor"

        // ============ 默认阈值 ============

        /**
         * 微笑检测死区阈值
         * 特征变化量小于此值时认为是中性表情，消除抖动
         * 数值越小，越容易检测到微笑
         */
        private const val SMILE_DEAD_ZONE = 0.003f

        /**
         * 撇嘴检测死区阈值
         * 数值越小，越容易检测到撇嘴
         */
        private const val FROWN_DEAD_ZONE = 0.002f

        /**
         * 最大微笑/撇嘴特征值（用于归一化）
         */
        private const val DEFAULT_SMILE_MAX = 0.06f
        private const val DEFAULT_FROWN_MAX = 0.04f

    }
    // ============ 校准数据 ============
    /**
     * 微笑校准数据
     */
    data class SmileCalibrationData(
        // 累积统计
        var sumLeftFeature: Float = 0f,
        var sumRightFeature: Float = 0f,
        var sumSquaredLeftFeature: Float = 0f,
        var sumSquaredRightFeature: Float = 0f,
        var frameCount: Int = 0,
        
        // 校准结果：中性表情时的特征基线
        var baselineLeftFeature: Float = 0f,
        var baselineRightFeature: Float = 0f,
        var stdLeftFeature: Float = 0f,
        var stdRightFeature: Float = 0f,
        
        // 观察到的范围（用于动态扩展）
        var observedSmileMax: Float = DEFAULT_SMILE_MAX,
        var observedFrownMax: Float = DEFAULT_FROWN_MAX,
        
        var isCalibrated: Boolean = false
    ) {
        /**
         * 添加一帧特征数据
         */
        fun addFrame(leftFeature: Float, rightFeature: Float) {
            sumLeftFeature += leftFeature
            sumRightFeature += rightFeature
            sumSquaredLeftFeature += leftFeature * leftFeature
            sumSquaredRightFeature += rightFeature * rightFeature
            frameCount++
        }
        
        /**
         * 完成校准
         */
        fun finishCalibration() {
            if (frameCount < 10) {
                Log.w(TAG, "Too few frames for smile calibration: $frameCount")
                return
            }
            
            // 计算平均值作为基线
            baselineLeftFeature = sumLeftFeature / frameCount
            baselineRightFeature = sumRightFeature / frameCount
            
            // 计算标准差
            val varianceLeft = (sumSquaredLeftFeature / frameCount) - (baselineLeftFeature * baselineLeftFeature)
            val varianceRight = (sumSquaredRightFeature / frameCount) - (baselineRightFeature * baselineRightFeature)
            stdLeftFeature = sqrt(max(0f, varianceLeft))
            stdRightFeature = sqrt(max(0f, varianceRight))
            
            isCalibrated = true
            
            Log.i(TAG, String.format(
                "😊 Smile calibrated: baseline L=%.4f R=%.4f, std L=%.4f R=%.4f",
                baselineLeftFeature, baselineRightFeature, stdLeftFeature, stdRightFeature
            ))
        }
        
        /**
         * 动态更新观察范围（只扩大不缩小）
         */
        fun updateObservedRange(leftDelta: Float, rightDelta: Float) {
            val avgDelta = (leftDelta + rightDelta) / 2f
            
            if (avgDelta > 0) {
                // 微笑方向
                val newMax = avgDelta * 1.2f  // 留 20% 余量
                if (newMax > observedSmileMax) {
                    observedSmileMax = newMax.coerceAtMost(0.15f)  // 上限
                }
            } else if (avgDelta < 0) {
                // 撇嘴方向
                val newMax = abs(avgDelta) * 1.2f
                if (newMax > observedFrownMax) {
                    observedFrownMax = newMax.coerceAtMost(0.10f)  // 上限
                }
            }
        }
        
        /**
         * 重置
         */
        fun reset() {
            sumLeftFeature = 0f
            sumRightFeature = 0f
            sumSquaredLeftFeature = 0f
            sumSquaredRightFeature = 0f
            frameCount = 0
            baselineLeftFeature = 0f
            baselineRightFeature = 0f
            stdLeftFeature = 0f
            stdRightFeature = 0f
            observedSmileMax = DEFAULT_SMILE_MAX
            observedFrownMax = DEFAULT_FROWN_MAX
            isCalibrated = false
        }
    }
    
    val calibration = SmileCalibrationData()
    
    /**
     * 校准帧数
     */
    var calibrationFrames: Int = FRAME_COUNT_CALIBRATE
    
    /**
     * 是否正在校准期
     */
    val isCalibrating: Boolean
        get() = !calibration.isCalibrated
    
    // ============ 俯仰角补偿 ============
    
    /**
     * 校准时的头部姿态
     */
    private var calibrationPitch: Float = 0f
    private var calibrationYaw: Float = 0f
    private var calibrationRoll: Float = 0f
    
    /**
     * 是否启用俯仰角补偿
     */
    var enablePitchCompensation: Boolean = true
    
    /**
     * 俯仰角补偿阈值（度）
     */
    var pitchCompensationThreshold: Float = 5f
    
    /**
     * 俯仰角补偿系数
     * 经验值：每度俯仰对微笑特征的影响
     */
    var pitchCompensationFactor: Float = 0.0008f
    
    /**
     * Roll 角补偿系数
     * 当头部倾斜时，左右嘴角高度差会变化
     */
    var rollCompensationFactor: Float = 0.001f
    
    /**
     * 设置校准时的头部姿态
     */
    fun setCalibrationPose(pitch: Float, yaw: Float, roll: Float) {
        calibrationPitch = pitch
        calibrationYaw = yaw
        calibrationRoll = roll
        Log.i(TAG, String.format(
            "📐 Smile calibration pose: pitch=%.1f°, yaw=%.1f°, roll=%.1f°",
            pitch, yaw, roll
        ))
    }
    
    /**
     * 计算俯仰角补偿
     */
    private fun calculatePitchCompensation(currentPitch: Float): Float {
        val deltaPitch = currentPitch - calibrationPitch
        if (abs(deltaPitch) <= pitchCompensationThreshold) {
            return 0f
        }
        
        // 低头时嘴角看起来更低，需要正向补偿
        val effectiveDelta = if (deltaPitch > 0) {
            deltaPitch - pitchCompensationThreshold
        } else {
            deltaPitch + pitchCompensationThreshold
        }
        
        return effectiveDelta * pitchCompensationFactor
    }
    
    /**
     * 计算 Roll 角补偿（左右嘴角差异）
     * @return Pair(leftCompensation, rightCompensation)
     */
    private fun calculateRollCompensation(currentRoll: Float): Pair<Float, Float> {
        val deltaRoll = currentRoll - calibrationRoll
        if (abs(deltaRoll) <= 3f) {
            return Pair(0f, 0f)
        }
        
        // 当头向右倾斜 (roll > 0)，左嘴角看起来更高，右嘴角看起来更低
        val compensation = deltaRoll * rollCompensationFactor
        return Pair(-compensation, compensation)
    }

    fun extractSmileBlendShapes(
        keyPoints: FaceKeyPoints,
        blendShapes: ARKitBlendShapes,
        headPose: HeadPose? = null,
        landmarks: List<Point3D>? = null
    ) {
        // 使用嘴巴外轮廓点
        val mouthOuter = keyPoints.mouthOuter
        if (mouthOuter.size < 16) {
            Log.w(TAG, "Not enough mouth points: ${mouthOuter.size}")
            return
        }
        
        // 获取面部高度用于归一化
        val faceHeight = keyPoints.faceHeight
        if (faceHeight < 1e-6f) {
            return
        }
        
        // 1. 提取原始微笑特征
        val (rawLeftFeature, rawRightFeature) = extractSmileFeatures(mouthOuter, faceHeight)
        
        // 2. 应用俯仰角和 Roll 角补偿
        var compensatedLeftFeature = rawLeftFeature
        var compensatedRightFeature = rawRightFeature
        
        if (enablePitchCompensation && headPose != null && calibration.isCalibrated) {
            val pitchComp = calculatePitchCompensation(headPose.pitch)
            compensatedLeftFeature += pitchComp
            compensatedRightFeature += pitchComp
            
            val (rollCompLeft, rollCompRight) = calculateRollCompensation(headPose.roll)
            compensatedLeftFeature += rollCompLeft
            compensatedRightFeature += rollCompRight
        }
        
        // 3. 校准期收集数据（使用原始特征）
        if (!calibration.isCalibrated) {
            calibration.addFrame(rawLeftFeature, rawRightFeature)
            if (calibration.frameCount >= calibrationFrames) {
                calibration.finishCalibration()
                // 设置校准姿态
                if (headPose != null) {
                    setCalibrationPose(headPose.pitch, headPose.yaw, headPose.roll)
                }
            }
        }
        
        // 4. 计算相对于基线的变化量
        val leftDelta = compensatedLeftFeature - calibration.baselineLeftFeature
        val rightDelta = compensatedRightFeature - calibration.baselineRightFeature
        
        // 5. 动态更新范围
        if (calibration.isCalibrated) {
            calibration.updateObservedRange(leftDelta, rightDelta)
        }
        
        // 6. 转换为 BlendShape 值
        val (smileLeft, frownLeft) = featureDeltaToBlendShape(
            leftDelta, 
            calibration.observedSmileMax, 
            calibration.observedFrownMax
        )
        val (smileRight, frownRight) = featureDeltaToBlendShape(
            rightDelta, 
            calibration.observedSmileMax, 
            calibration.observedFrownMax
        )
        
        // 7. 输出
        blendShapes.mouthSmileLeft = smileLeft
        blendShapes.mouthSmileRight = smileRight
        blendShapes.mouthFrownLeft = frownLeft
        blendShapes.mouthFrownRight = frownRight
    }
    
    /**
     * 从嘴巴轮廓提取微笑特征
     * 
     * 特征 = (嘴唇中心Y - 嘴角Y) / faceHeight
     * 正值 = 嘴角高于中心 = 微笑
     * 负值 = 嘴角低于中心 = 撇嘴
     * 
     * @return Pair(leftFeature, rightFeature)
     */
    private fun extractSmileFeatures(mouthOuter: List<Point3D>, faceHeight: Float): Pair<Float, Float> {
        // mouthOuter 索引：
        // [0] = 左嘴角
        // [5] = 下唇中心
        // [10] = 右嘴角
        // [15] = 上唇中心
        
        val leftCorner = mouthOuter[0]
        val rightCorner = mouthOuter[10]
        val upperLipCenter = mouthOuter[15]
        val lowerLipCenter = mouthOuter[5]
        
        // 嘴唇中心（上下唇的平均）
        val mouthCenterY = (upperLipCenter.y + lowerLipCenter.y) / 2f
        
        // 归一化的特征值
        // 注意：图像坐标系 Y 向下，所以嘴角 Y < 中心 Y 表示嘴角更高（微笑）
        val leftFeature = (mouthCenterY - leftCorner.y) / faceHeight
        val rightFeature = (mouthCenterY - rightCorner.y) / faceHeight
        
        return Pair(leftFeature, rightFeature)
    }
    
    /**
     * 将特征变化量转换为 smile/frown BlendShape
     * 
     * @param delta 相对于基线的变化量
     * @param smileMax 微笑最大值（用于归一化）
     * @param frownMax 撇嘴最大值
     * @return Pair(smile, frown)，均为 [0, 1]
     */
    private fun featureDeltaToBlendShape(
        delta: Float,
        smileMax: Float,
        frownMax: Float
    ): Pair<Float, Float> {
        return when {
            delta > SMILE_DEAD_ZONE -> {
                // 微笑
                val smile = ((delta - SMILE_DEAD_ZONE) / (smileMax - SMILE_DEAD_ZONE)).coerceIn(0f, 1f)
                Pair(smile, 0f)
            }
            delta < -FROWN_DEAD_ZONE -> {
                // 撇嘴
                val frown = ((abs(delta) - FROWN_DEAD_ZONE) / (frownMax - FROWN_DEAD_ZONE)).coerceIn(0f, 1f)
                Pair(0f, frown)
            }
            else -> {
                // 中性
                Pair(0f, 0f)
            }
        }
    }
    
    /**
     * 重置校准
     */
    fun resetCalibration() {
        calibration.reset()
        calibrationPitch = 0f
        calibrationYaw = 0f
        calibrationRoll = 0f
        Log.i(TAG, "🔄 Smile calibration reset")
    }

    fun reset() {
        resetCalibration()
    }
}