package com.live2d.facecapture

import com.example.commondata.Point3D
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 面部追踪置信度检测器
 * 
 * 复刻 MediaPipe 的追踪置信度设计，用于判断：
 * 1. 是否需要重新运行人脸检测
 * 2. 当前追踪是否可信
 * 
 * ==================== MediaPipe 原理分析 ====================
 * 
 * MediaPipe 的追踪置信度来自两个来源：
 * 
 * 1. **Presence Score** (存在分数)
 *    - 来源：Landmark 模型的输出之一
 *    - 模型输出两个 tensor：landmarks tensor + presence_flag tensor
 *    - presence_flag 经过 Sigmoid 激活后得到 [0, 1] 的置信度
 *    - 与 `min_detection_confidence` 阈值比较
 * 
 * 2. **Association Similarity** (关联相似度)
 *    - 使用 `AssociationNormRectCalculator` 计算
 *    - 比较上一帧的人脸框和当前帧的人脸框
 *    - 基于 IoU (Intersection over Union) 计算相似度
 *    - 与 `min_tracking_confidence` (即 min_similarity_threshold) 比较
 * 
 * ==================== 复刻策略 ====================
 * 
 * 由于我们无法直接获取模型的 presence_score，使用以下替代方案：
 * 
 * 1. **基于 IoU 的框相似度**
 *    - 比较连续帧的人脸框位置
 *    - IoU > threshold 表示追踪稳定
 * 
 * 2. **基于 Landmark 的稳定性**
 *    - 计算连续帧关键点的移动量
 *    - 移动量过大表示追踪可能丢失
 * 
 * 3. **基于面部几何的有效性**
 *    - 检查关键点的几何关系是否合理
 *    - 例如：眼睛应该在鼻子上方
 * 
 * @author yrzhu
 * @date 2025/1/15
 */
class TrackingConfidenceDetector {
    companion object {
        private const val TAG = "TrackingConfidenceDetector"
    }

    // ============ 配置参数 ============
    
    /**
     * 最小追踪置信度阈值 (对应 MediaPipe 的 min_tracking_confidence)
     * 低于此值会触发重新检测
     */
    var minTrackingConfidence: Float = 0.5f
    
    /**
     * IoU 阈值：框重叠度低于此值认为追踪丢失
     */
    var minIoUThreshold: Float = 0.3f
    
    /**
     * 最大允许的关键点移动量 (归一化坐标)
     * 超过此值认为追踪不稳定
     */
    var maxLandmarkMovement: Float = 0.15f
    
    /**
     * 平滑系数：用于平滑置信度变化
     */
    var smoothingFactor: Float = 0.7f
    
    // ============ 状态 ============
    
    private var lastFaceBox: FaceBox? = null
    private var lastLandmarks: List<Point3D>? = null
    private var smoothedConfidence: Float = 1.0f
    private var framesSinceLastDetection: Int = 0
    
    /**
     * 人脸框数据类
     */
    data class FaceBox(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    ) {
        val width: Float get() = right - left
        val height: Float get() = bottom - top
        val centerX: Float get() = (left + right) / 2f
        val centerY: Float get() = (top + bottom) / 2f
        val area: Float get() = width * height
    }

/**
 * 追踪状态结果
 */
data class TrackingStatus(
    val confidence: Float,          // 综合置信度 [0, 1]
    val presenceScore: Float,       // 模型输出的存在分数 [0, 1]
    val iouScore: Float,            // IoU 相似度 [0, 1]
    val isTrackingValid: Boolean,   // 是否追踪有效
    val shouldRedetect: Boolean,    // 是否应该重新检测
    val reason: String              // 状态原因（调试用）
)
    
    /**
     * 计算两个矩形的 IoU (Intersection over Union)
     */
    fun calculateIoU(box1: FaceBox, box2: FaceBox): Float {
        // 计算交集
        val intersectLeft = max(box1.left, box2.left)
        val intersectTop = max(box1.top, box2.top)
        val intersectRight = min(box1.right, box2.right)
        val intersectBottom = min(box1.bottom, box2.bottom)
        
        val intersectWidth = max(0f, intersectRight - intersectLeft)
        val intersectHeight = max(0f, intersectBottom - intersectTop)
        val intersectArea = intersectWidth * intersectHeight
        
        // 计算并集
        val unionArea = box1.area + box2.area - intersectArea
        
        return if (unionArea > 0f) {
            intersectArea / unionArea
        } else {
            0f
        }
    }
    
    /**
     * 计算关键点的平均移动量
     * 
     * @param prevLandmarks 上一帧的关键点
     * @param currLandmarks 当前帧的关键点
     * @return 归一化的平均移动量
     */
    fun calculateLandmarkMovement(
        prevLandmarks: List<Point3D>,
        currLandmarks: List<Point3D>
    ): Float {
        if (prevLandmarks.size != currLandmarks.size || prevLandmarks.isEmpty()) {
            return Float.MAX_VALUE
        }
        
        var totalMovement = 0f
        for (i in prevLandmarks.indices) {
            val dx = currLandmarks[i].x - prevLandmarks[i].x
            val dy = currLandmarks[i].y - prevLandmarks[i].y
            totalMovement += sqrt(dx * dx + dy * dy)
        }
        
        return totalMovement / prevLandmarks.size
    }
    
    /**
     * 检查关键点几何有效性
     * 使用简单的启发式规则检查关键点是否合理
     * 
     * @return 有效性分数 [0, 1]
     */
    fun checkLandmarkValidity(landmarks: List<Point3D>): Float {
        if (landmarks.size < 468) {
            return 0f
        }
        
        var validityScore = 1.0f
        
        // 检查 1: 左右眼应该在鼻子两侧
        // 使用 MediaPipe 索引
        val leftEyeCenter = landmarks[468]  // 左眼虹膜中心 (如果有)
        val rightEyeCenter = landmarks[473] // 右眼虹膜中心 (如果有)
        val noseTip = landmarks[1]          // 鼻尖
        
        // 检查眼睛是否在鼻子上方
        if (leftEyeCenter.y > noseTip.y || rightEyeCenter.y > noseTip.y) {
            validityScore *= 0.5f  // 惩罚
        }
        
        // 检查 2: 嘴巴应该在鼻子下方
        val upperLip = landmarks[13]  // 上唇中心
        if (upperLip.y < noseTip.y) {
            validityScore *= 0.5f
        }
        
        // 检查 3: 左右眼应该有合理的间距
        val eyeDistance = abs(leftEyeCenter.x - rightEyeCenter.x)
        if (eyeDistance < 0.05f || eyeDistance > 0.5f) {
            validityScore *= 0.7f
        }
        
        return validityScore
    }
    
    /**
     * 从 landmarks 计算人脸框
     */
    fun calculateFaceBox(landmarks: List<Point3D>): FaceBox {
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
        
        return FaceBox(minX, minY, maxX, maxY)
    }
    
    /**
     * 更新追踪状态并返回结果
     * 
     * @param currentLandmarks 当前帧的关键点（归一化坐标）
     * @param presenceScore 模型输出的存在分数 [0, 1]，来自 Landmark 模型
     * @param currentFaceBox 当前帧的人脸框（可选，如果不提供会从 landmarks 计算）
     * @return 追踪状态
     */
    fun updateAndCheck(
        currentLandmarks: List<Point3D>,
        presenceScore: Float = 1.0f,
        currentFaceBox: FaceBox? = null
    ): TrackingStatus {
        val box = currentFaceBox ?: calculateFaceBox(currentLandmarks)
        
        // 首帧处理
        if (lastFaceBox == null || lastLandmarks == null) {
            lastFaceBox = box
            lastLandmarks = currentLandmarks.toList()
            framesSinceLastDetection = 0
            smoothedConfidence = presenceScore
            
            return TrackingStatus(
                confidence = presenceScore,
                presenceScore = presenceScore,
                iouScore = 1.0f,
                isTrackingValid = presenceScore >= minTrackingConfidence,
                shouldRedetect = presenceScore < minTrackingConfidence,
                reason = "First frame, presence=${"%.2f".format(presenceScore)}"
            )
        }
        
        framesSinceLastDetection++
        
        // 计算各项指标
        val iou = calculateIoU(lastFaceBox!!, box)
        val movement = calculateLandmarkMovement(lastLandmarks!!, currentLandmarks)
        val validity = checkLandmarkValidity(currentLandmarks)
        
        // 计算综合置信度
        // 使用 MediaPipe 的方式：IoU 用于框匹配，presenceScore 用于存在判断
        val iouScore = iou  // 直接使用 IoU 作为相似度
        val movementScore = (1f - movement / maxLandmarkMovement).coerceIn(0f, 1f)
        
        // 加权综合：presenceScore 权重最高（模型输出），其次是 IoU
        val rawConfidence = (presenceScore * 0.5f + iouScore * 0.3f + movementScore * 0.2f)
            .coerceIn(0f, 1f)
        
        // 平滑
        smoothedConfidence = smoothingFactor * smoothedConfidence + 
                            (1f - smoothingFactor) * rawConfidence
        
        // 判断是否需要重新检测
        // 使用 MediaPipe 的逻辑：
        // 1. presenceScore 低于阈值 → 人脸可能不存在
        // 2. IoU 低于阈值 → 追踪可能丢失
        // 3. 周期性强制检测
        val isValid = presenceScore >= minTrackingConfidence && 
                     iou >= minIoUThreshold
        val shouldRedetect = !isValid || 
                            framesSinceLastDetection > 30 ||
                            iou < 0.1f ||
                            presenceScore < 0.3f
        
        // 构建原因字符串（调试用）
        val reason = buildString {
            append("Pres=%.2f ".format(presenceScore))
            append("IoU=%.2f ".format(iou))
            append("Move=%.3f ".format(movement))
            append("Conf=%.2f".format(smoothedConfidence))
        }
        
        // 更新状态
        lastFaceBox = box
        lastLandmarks = currentLandmarks.toList()
        
        if (shouldRedetect) {
            framesSinceLastDetection = 0
        }
        
        return TrackingStatus(
            confidence = smoothedConfidence,
            presenceScore = presenceScore,
            iouScore = iou,
            isTrackingValid = isValid,
            shouldRedetect = shouldRedetect,
            reason = reason
        )
    }
    
    /**
     * 通知检测器已执行了人脸检测
     * 在重新检测后调用，重置计数器
     */
    fun onDetectionPerformed() {
        framesSinceLastDetection = 0
        smoothedConfidence = 1.0f
    }
    
    /**
     * 重置状态
     */
    fun reset() {
        lastFaceBox = null
        lastLandmarks = null
        smoothedConfidence = 1.0f
        framesSinceLastDetection = 0
    }
}
