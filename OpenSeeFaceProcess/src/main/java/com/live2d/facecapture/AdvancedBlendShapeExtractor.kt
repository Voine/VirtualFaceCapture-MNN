/**
 * 高级 BlendShape 提取器
 * 
 * 基于 MediaPipe 478 点位提取 ARKit 剩余的 BlendShape 参数
 * 包括：眉毛、脸颊、下巴、鼻子、以及更多嘴部表情
 * 
 * MediaPipe 478 点位说明：
 * - 0-467: 标准面部网格点
 * - 468-472: 左虹膜 (468=中心, 469-472=边缘)
 * - 473-477: 右虹膜 (473=中心, 474-477=边缘)
 */

package com.live2d.facecapture

import android.util.Log
import com.example.commondata.ARKitBlendShapes
import com.example.commondata.FaceLandmarkIndices
import com.example.commondata.Point3D

/**
 * 高级 BlendShape 提取器
 */
class AdvancedBlendShapeExtractor {
    /**
     * 眉毛参数
     */
    object BrowParams {
        // 眉毛上下移动的灵敏度
        var browUpSensitivity: Float = 2.5f
        var browDownSensitivity: Float = 3.0f
        
        // 阈值 (相对于面部高度的比例)
        var browUpThreshold: Float = 0.01f
        var browDownThreshold: Float = 0.01f
    }
    
    /**
     * 脸颊参数
     */
    object CheekParams {
        var cheekPuffSensitivity: Float = 3.0f
        var cheekSquintSensitivity: Float = 2.5f
    }
    
    /**
     * 下巴参数
     */
    object JawParams {
        var jawMoveSensitivity: Float = 3.0f
        var jawMoveThreshold: Float = 0.01f
    }
    
    /**
     * 鼻子参数
     */
    object NoseParams {
        var sneerSensitivity: Float = 4.0f
    }
    
    /**
     * 嘴部高级参数
     */
    object MouthAdvancedParams {
        var lipRollSensitivity: Float = 3.0f
        var lipStretchSensitivity: Float = 2.5f
        var mouthMoveSensitivity: Float = 3.0f
    }
    
    // ==================== 基线缓存 (用于相对位移计算) ====================
    
    private var baselineBrowHeight: Float = 0f
    private var baselineCheekWidth: Float = 0f
    private var baselineNoseWidth: Float = 0f
    private var isBaselineSet: Boolean = false
    
    /**
     * 设置基线
     * 在用户保持中性表情时调用，记录各特征的基准位置
     */
    fun calibrate(landmarks: List<Point3D>, faceHeight: Float, faceWidth: Float) {
        if (landmarks.size < 478) return
        
        // 记录眉毛到眼睛的基准距离
        val leftBrowY = landmarks[FaceLandmarkIndices.ImageRightEyebrow.MIDDLE].y
        val leftEyeY = landmarks[FaceLandmarkIndices.ImageRightEye.UPPER_EYELID[3]].y
        baselineBrowHeight = (leftEyeY - leftBrowY) / faceHeight
        
        // 记录脸颊基准宽度
        val leftCheek = landmarks[FaceLandmarkIndices.Cheek.IMAGE_LEFT[0]]
        val rightCheek = landmarks[FaceLandmarkIndices.Cheek.IMAGE_RIGHT[0]]
        baselineCheekWidth = leftCheek.distance2DTo(rightCheek) / faceWidth
        
        // 记录鼻翼基准宽度
        val leftNose = landmarks[FaceLandmarkIndices.Nose.IMAGE_LEFT_WING]
        val rightNose = landmarks[FaceLandmarkIndices.Nose.IMAGE_RIGHT_WING]
        baselineNoseWidth = leftNose.distance2DTo(rightNose) / faceWidth
        
        isBaselineSet = true
        
        Log.i("AdvancedBlendShape",
            "✅ Calibrated: browHeight=%.4f, cheekWidth=%.4f, noseWidth=%.4f".format(
                baselineBrowHeight, baselineCheekWidth, baselineNoseWidth
            ))
    }
    
    /**
     * 重置基线
     */
    fun reset() {
        baselineBrowHeight = 0f
        baselineCheekWidth = 0f
        baselineNoseWidth = 0f
        isBaselineSet = false
    }
    
    // ==================== BlendShape 提取方法 ====================
    
    /**
     * 提取所有高级 BlendShape
     * 
     * @param landmarks 478 个关键点
     * @param blendShapes 输出的 BlendShape 对象
     * @param faceHeight 面部高度 (用于归一化)
     * @param faceWidth 面部宽度 (用于归一化)
     */
    fun extractAdvancedBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceHeight: Float,
        faceWidth: Float
    ) {
        if (landmarks.size < 468) {
            Log.w("AdvancedBlendShape", "Insufficient landmarks: ${landmarks.size}")
            return
        }
        
        // 1. 眉毛 BlendShape
        extractBrowBlendShapes(landmarks, blendShapes, faceHeight)
        
        // 2. 脸颊 BlendShape
        extractCheekBlendShapes(landmarks, blendShapes, faceWidth)
        
        // 3. 下巴 BlendShape (jawLeft, jawRight, jawForward)
        extractJawBlendShapes(landmarks, blendShapes, faceWidth)
        
        // 4. 鼻子 BlendShape
        extractNoseBlendShapes(landmarks, blendShapes, faceWidth)
        
        // 5. 嘴部高级 BlendShape
        extractMouthAdvancedBlendShapes(landmarks, blendShapes, faceWidth, faceHeight)
    }
    
    /**
     * 提取眉毛 BlendShape
     * 
     * - browDownLeft/Right: 眉毛下压 (皱眉)
     * - browInnerUp: 眉毛内侧上扬 (惊讶/担忧)
     * - browOuterUpLeft/Right: 眉毛外侧上扬
     */
    private fun extractBrowBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceHeight: Float
    ) {
        // 获取关键点
        val leftBrowInner = landmarks[FaceLandmarkIndices.ImageRightEyebrow.INNER]
        val leftBrowMiddle = landmarks[FaceLandmarkIndices.ImageRightEyebrow.MIDDLE]
        val leftBrowOuter = landmarks[FaceLandmarkIndices.ImageRightEyebrow.OUTER]
        
        val rightBrowInner = landmarks[FaceLandmarkIndices.ImageLeftEyebrow.INNER]
        val rightBrowMiddle = landmarks[FaceLandmarkIndices.ImageLeftEyebrow.MIDDLE]
        val rightBrowOuter = landmarks[FaceLandmarkIndices.ImageLeftEyebrow.OUTER]
        
        // 参考点
        val noseRoot = landmarks[FaceLandmarkIndices.Nose.ROOT]
        val leftEyeTop = landmarks[FaceLandmarkIndices.ImageRightEye.UPPER_EYELID[3]]
        val rightEyeTop = landmarks[FaceLandmarkIndices.ImageLeftEye.UPPER_EYELID[3]]
        
        // 计算眉毛到眼睛的距离 (归一化)
        val leftBrowToEye = (leftEyeTop.y - leftBrowMiddle.y) / faceHeight
        val rightBrowToEye = (rightEyeTop.y - rightBrowMiddle.y) / faceHeight
        
        // 基准值 (如果没有校准，使用默认值)
        val baseline = if (isBaselineSet) baselineBrowHeight else 0.08f
        
        // browDown: 眉毛下压 (眉眼距离变小)
        val leftBrowDownOffset = baseline - leftBrowToEye
        val rightBrowDownOffset = baseline - rightBrowToEye
        
        blendShapes.browDownLeft = (leftBrowDownOffset * BrowParams.browDownSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        blendShapes.browDownRight = (rightBrowDownOffset * BrowParams.browDownSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        
        // browInnerUp: 眉毛内侧上扬 (计算内侧相对于鼻根的位移)
        val browInnerY = (leftBrowInner.y + rightBrowInner.y) / 2f
        val innerUpOffset = (noseRoot.y - browInnerY) / faceHeight - 0.05f  // 相对于中性位置
        blendShapes.browInnerUp = (innerUpOffset * BrowParams.browUpSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        
        // browOuterUp: 眉毛外侧上扬
        val leftOuterUp = leftBrowToEye - baseline
        val rightOuterUp = rightBrowToEye - baseline
        
        blendShapes.browOuterUpLeft = (leftOuterUp * BrowParams.browUpSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        blendShapes.browOuterUpRight = (rightOuterUp * BrowParams.browUpSensitivity / 0.02f)
            .coerceIn(0f, 1f)
    }
    
    /**
     * 提取脸颊 BlendShape
     * 
     * - cheekPuff: 鼓腮
     * - cheekSquintLeft/Right: 脸颊挤压 (眯眼笑时)
     */
    private fun extractCheekBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceWidth: Float
    ) {
        // cheekPuff: 计算脸颊区域的外扩程度
        // 原理：鼓腮时，脸颊点向外移动，左右脸颊距离增大
        val leftCheekPoints = FaceLandmarkIndices.Cheek.IMAGE_LEFT.map { landmarks[it] }
        val rightCheekPoints = FaceLandmarkIndices.Cheek.IMAGE_RIGHT.map { landmarks[it] }
        
        // 计算脸颊中心点
        val leftCheekCenter = Point3D(
            leftCheekPoints.map { it.x }.average().toFloat(),
            leftCheekPoints.map { it.y }.average().toFloat(),
            leftCheekPoints.map { it.z }.average().toFloat()
        )
        val rightCheekCenter = Point3D(
            rightCheekPoints.map { it.x }.average().toFloat(),
            rightCheekPoints.map { it.y }.average().toFloat(),
            rightCheekPoints.map { it.z }.average().toFloat()
        )
        
        val cheekDistance = leftCheekCenter.distance2DTo(rightCheekCenter) / faceWidth
        val cheekBaseline = if (isBaselineSet) baselineCheekWidth else 0.8f
        
        val puffOffset = cheekDistance - cheekBaseline
        blendShapes.cheekPuff = (puffOffset * CheekParams.cheekPuffSensitivity / 0.05f)
            .coerceIn(0f, 1f)
        
        // cheekSquint: 计算脸颊上部与下眼睑的距离
        // 原理：眯眼笑时，脸颊上移，距离下眼睑更近
        val leftCheekSquint = landmarks[FaceLandmarkIndices.Cheek.IMAGE_LEFT_SQUINT]
        val leftLowerEyelid = landmarks[FaceLandmarkIndices.Cheek.IMAGE_LEFT_LOWER_EYELID]
        val rightCheekSquint = landmarks[FaceLandmarkIndices.Cheek.IMAGE_RIGHT_SQUINT]
        val rightLowerEyelid = landmarks[ FaceLandmarkIndices.Cheek.IMAGE_RIGHT_LOWER_EYELID]
        
        val leftSquintDist = leftLowerEyelid.distance2DTo(leftCheekSquint) / faceWidth
        val rightSquintDist = rightLowerEyelid.distance2DTo(rightCheekSquint) / faceWidth
        
        // 距离越小，squint 越大
        val baselineSquintDist = 0.08f  // 正常时的距离
        blendShapes.cheekSquintLeft = ((baselineSquintDist - leftSquintDist) * CheekParams.cheekSquintSensitivity / 0.03f)
            .coerceIn(0f, 1f)
        blendShapes.cheekSquintRight = ((baselineSquintDist - rightSquintDist) * CheekParams.cheekSquintSensitivity / 0.03f)
            .coerceIn(0f, 1f)
    }
    
    /**
     * 提取下巴 BlendShape (除 jawOpen 外)
     * 
     * - jawLeft: 下巴向左移动
     * - jawRight: 下巴向右移动
     * - jawForward: 下巴前突
     */
    private fun extractJawBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceWidth: Float
    ) {
        val chinCenter = landmarks[FaceLandmarkIndices.Jaw.CHIN_CENTER]
        val mouthCenter = landmarks[FaceLandmarkIndices.Mouth.UPPER_LIP_CENTER]
        val noseRoot = landmarks[FaceLandmarkIndices.Nose.ROOT]
        
        // 计算面部中线 (鼻根到嘴中心)
        val faceCenterX = (noseRoot.x + mouthCenter.x) / 2f
        
        // jawLeft/Right: 下巴相对于面部中线的水平偏移
        val chinOffset = (chinCenter.x - faceCenterX) / faceWidth
        
        // 正偏移 = 下巴向右 (图像坐标系)
        // 注意：可能需要根据镜像情况调整
        blendShapes.jawRight = (chinOffset * JawParams.jawMoveSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        blendShapes.jawLeft = (-chinOffset * JawParams.jawMoveSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        
        // jawForward: 使用 Z 深度变化
        // 下巴前突时，Z 值变小 (更靠近相机)
        val baselineZ = noseRoot.z
        val chinZOffset = baselineZ - chinCenter.z
        blendShapes.jawForward = (chinZOffset * JawParams.jawMoveSensitivity / 0.05f)
            .coerceIn(0f, 1f)
    }
    
    /**
     * 提取鼻子 BlendShape
     * 
     * - noseSneerLeft/Right: 皱鼻 (厌恶表情)
     */
    private fun extractNoseBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceWidth: Float
    ) {
        // noseSneer: 鼻翼上提，鼻子皱纹区域收缩
        // 计算鼻翼相对于鼻尖的位置变化
        
        val noseTip = landmarks[FaceLandmarkIndices.Nose.TIP]
        val leftNoseWing = landmarks[FaceLandmarkIndices.Nose.IMAGE_LEFT_WING]
        val rightNoseWing = landmarks[FaceLandmarkIndices.Nose.IMAGE_RIGHT_WING]
        
        // 皱鼻时，鼻翼上移
        // 计算鼻翼 Y 坐标相对于鼻尖的变化
        val leftWingY = leftNoseWing.y - noseTip.y
        val rightWingY = rightNoseWing.y - noseTip.y
        
        // 同时考虑鼻翼区域的收缩
        val leftSneerArea = FaceLandmarkIndices.Nose.LEFT_SNEER_AREA.map { landmarks[it] }
        val rightSneerArea = FaceLandmarkIndices.Nose.RIGHT_SNEER_AREA.map { landmarks[it] }
        
        // 计算皱纹区域的 Z 深度变化 (皱鼻时凸起)
        val leftSneerZ = leftSneerArea.map { it.z }.average().toFloat()
        val rightSneerZ = rightSneerArea.map { it.z }.average().toFloat()
        val baselineZ = noseTip.z
        
        val leftSneerOffset = (baselineZ - leftSneerZ) + (-leftWingY * 0.5f)
        val rightSneerOffset = (baselineZ - rightSneerZ) + (-rightWingY * 0.5f)
        
        blendShapes.noseSneerLeft = (leftSneerOffset * NoseParams.sneerSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        blendShapes.noseSneerRight = (rightSneerOffset * NoseParams.sneerSensitivity / 0.02f)
            .coerceIn(0f, 1f)
    }
    
    /**
     * 提取嘴部高级 BlendShape
     * 
     * - mouthLeft/Right: 嘴巴整体左右移动
     * - mouthFunnel: 嘴唇向前突出成 O 形
     * - mouthPucker: 嘟嘴
     * - mouthRollLower/Upper: 嘴唇内卷
     * - mouthShrugLower/Upper: 嘴唇耸起
     * - mouthStretchLeft/Right: 嘴角向外拉伸
     * - mouthDimpleLeft/Right: 酒窝
     * - mouthPressLeft/Right: 嘴唇压紧
     * - mouthLowerDownLeft/Right: 下唇下拉
     * - mouthUpperUpLeft/Right: 上唇上提
     * - mouthClose: 嘴唇闭合 (有别于 jawOpen)
     */
    private fun extractMouthAdvancedBlendShapes(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceWidth: Float,
        faceHeight: Float
    ) {
        val leftCorner = landmarks[FaceLandmarkIndices.Mouth.IMAGE_LEFT_CORNER]
        val rightCorner = landmarks[FaceLandmarkIndices.Mouth.IMAGE_RIGHT_CORNER]
        val upperLipTop = landmarks[FaceLandmarkIndices.Mouth.UPPER_LIP_TOP]
        val upperLipCenter = landmarks[FaceLandmarkIndices.Mouth.UPPER_LIP_CENTER]
        val lowerLipCenter = landmarks[FaceLandmarkIndices.Mouth.LOWER_LIP_CENTER]
        val lowerLipBottom = landmarks[FaceLandmarkIndices.Mouth.LOWER_LIP_BOTTOM]
        
        // 嘴巴中心点
        val mouthCenterX = (leftCorner.x + rightCorner.x) / 2f
        val mouthCenterY = (upperLipCenter.y + lowerLipCenter.y) / 2f
        
        // 面部中线 X 坐标
        val noseTip = landmarks[FaceLandmarkIndices.Nose.TIP]
        val faceCenterX = noseTip.x
        
        // ===== mouthLeft/Right: 嘴巴整体偏移 =====
        val mouthOffsetX = (mouthCenterX - faceCenterX) / faceWidth
        blendShapes.mouthRight = (mouthOffsetX * MouthAdvancedParams.mouthMoveSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        blendShapes.mouthLeft = (-mouthOffsetX * MouthAdvancedParams.mouthMoveSensitivity / 0.02f)
            .coerceIn(0f, 1f)
        
        // ===== mouthFunnel/Pucker: 嘴唇前突 =====
        // 原理：发 "O" 音时，嘴唇 Z 值减小（向前突出）
        val mouthAvgZ = (upperLipCenter.z + lowerLipCenter.z + leftCorner.z + rightCorner.z) / 4f
        val mouthForwardOffset = noseTip.z - mouthAvgZ
        
        // 同时考虑嘴巴宽度（funnel 嘴巴收窄成 O 形）
        val mouthWidth = leftCorner.distance2DTo(rightCorner) / faceWidth
        val mouthWidthBaseline = 0.35f
        val widthNarrowing = mouthWidthBaseline - mouthWidth
        
        // funnel: 收窄 + 前突
        blendShapes.mouthFunnel = ((mouthForwardOffset * 0.5f + widthNarrowing) * 3f / 0.05f)
            .coerceIn(0f, 1f)
        
        // pucker: 更强的收窄（嘟嘴）
        blendShapes.mouthPucker = (widthNarrowing * 4f / 0.05f)
            .coerceIn(0f, 1f)
        
        // ===== mouthRollLower/Upper: 嘴唇内卷 =====
        // 原理：内卷时，唇外缘和唇内缘距离减小
        val upperLipHeight = (upperLipCenter.y - upperLipTop.y) / faceHeight
        val lowerLipHeight = (lowerLipBottom.y - lowerLipCenter.y) / faceHeight
        
        val lipHeightBaseline = 0.02f
        blendShapes.mouthRollUpper = ((lipHeightBaseline - upperLipHeight) * MouthAdvancedParams.lipRollSensitivity / 0.01f)
            .coerceIn(0f, 1f)
        blendShapes.mouthRollLower = ((lipHeightBaseline - lowerLipHeight) * MouthAdvancedParams.lipRollSensitivity / 0.01f)
            .coerceIn(0f, 1f)
        
        // ===== mouthShrugLower/Upper: 嘴唇耸起 =====
        // 下唇耸起时，下唇上移
        val lipOpenHeight = (lowerLipCenter.y - upperLipCenter.y) / faceHeight
        val lipOpenBaseline = 0.02f
        
        // 当嘴巴闭合且下唇上移时
        val shrugOffset = lipOpenBaseline - lipOpenHeight
        blendShapes.mouthShrugLower = (shrugOffset * 3f / 0.01f).coerceIn(0f, 1f)
        blendShapes.mouthShrugUpper = (shrugOffset * 2f / 0.01f).coerceIn(0f, 1f)
        
        // ===== mouthStretchLeft/Right: 嘴角向外拉伸 =====
        // 原理：露齿笑时嘴角向两侧拉伸
        val mouthWidthOffset = mouthWidth - mouthWidthBaseline
        blendShapes.mouthStretchLeft = (mouthWidthOffset * MouthAdvancedParams.lipStretchSensitivity / 0.03f)
            .coerceIn(0f, 1f)
        blendShapes.mouthStretchRight = (mouthWidthOffset * MouthAdvancedParams.lipStretchSensitivity / 0.03f)
            .coerceIn(0f, 1f)
        
        // ===== mouthDimpleLeft/Right: 酒窝 =====
        // 酒窝在笑时凹陷，Z 值增大
        val leftDimple = landmarks[FaceLandmarkIndices.Mouth.IMAGE_LEFT_DIMPLE]
        val rightDimple = landmarks[FaceLandmarkIndices.Mouth.IMAGE_RIGHT_DIMPLE]
        
        val dimpleDepth = 0.01f
        blendShapes.mouthDimpleLeft = ((leftDimple.z - leftCorner.z) * 5f / dimpleDepth)
            .coerceIn(0f, 1f)
        blendShapes.mouthDimpleRight = ((rightDimple.z - rightCorner.z) * 5f / dimpleDepth)
            .coerceIn(0f, 1f)
        
        // ===== mouthPressLeft/Right: 嘴唇压紧 =====
        // 嘴唇压紧时变薄
        val totalLipHeight = (lowerLipBottom.y - upperLipTop.y) / faceHeight
        val pressOffset = 0.06f - totalLipHeight
        blendShapes.mouthPressLeft = (pressOffset * 3f / 0.02f).coerceIn(0f, 1f)
        blendShapes.mouthPressRight = (pressOffset * 3f / 0.02f).coerceIn(0f, 1f)
        
        // ===== mouthLowerDownLeft/Right: 下唇下拉 =====
        val lowerLipDrop = (lowerLipBottom.y - mouthCenterY) / faceHeight
        val dropBaseline = 0.03f
        val dropOffset = lowerLipDrop - dropBaseline
        
        // 左右分开计算
        val leftLowerLip = FaceLandmarkIndices.Mouth.LOWER_LIP_IMAGE_LEFT.map { landmarks[it] }
        val rightLowerLip =  FaceLandmarkIndices.Mouth.LOWER_LIP_IMAGE_RIGHT.map { landmarks[it] }
        val leftLowerY = leftLowerLip.map { it.y }.average().toFloat()
        val rightLowerY = rightLowerLip.map { it.y }.average().toFloat()
        
        blendShapes.mouthLowerDownLeft = ((leftLowerY - mouthCenterY) / faceHeight * 3f / 0.03f)
            .coerceIn(0f, 1f)
        blendShapes.mouthLowerDownRight = ((rightLowerY - mouthCenterY) / faceHeight * 3f / 0.03f)
            .coerceIn(0f, 1f)
        
        // ===== mouthUpperUpLeft/Right: 上唇上提 =====
        val leftUpperLip = FaceLandmarkIndices.Mouth.UPPER_LIP_IMAGE_LEFT.map { landmarks[it] }
        val rightUpperLip = FaceLandmarkIndices.Mouth.UPPER_LIP_IMAGE_RIGHT.map { landmarks[it] }
        val leftUpperY = leftUpperLip.map { it.y }.average().toFloat()
        val rightUpperY = rightUpperLip.map { it.y }.average().toFloat()
        
        blendShapes.mouthUpperUpLeft = ((mouthCenterY - leftUpperY) / faceHeight * 3f / 0.03f)
            .coerceIn(0f, 1f)
        blendShapes.mouthUpperUpRight = ((mouthCenterY - rightUpperY) / faceHeight * 3f / 0.03f)
            .coerceIn(0f, 1f)
        
        // ===== mouthClose: 嘴唇闭合 =====
        // 与 jawOpen 不同，mouthClose 是嘴唇主动闭合
        // 当 jawOpen > 0 但上下唇距离小时，mouthClose > 0
        val lipDistance = lowerLipCenter.distance2DTo(upperLipCenter) / faceHeight
        val closeThreshold = 0.01f
        blendShapes.mouthClose = ((closeThreshold - lipDistance) * 5f / closeThreshold)
            .coerceIn(0f, 1f)
    }
    
    /**
     * 提取眼部挤压 BlendShape (eyeSquint)
     * 
     * 这个通常与 smile 和 cheekSquint 相关
     * - eyeSquintLeft/Right: 眼睛挤压 (微笑时的眯眼)
     */
    fun extractEyeSquint(
        landmarks: List<Point3D>,
        blendShapes: ARKitBlendShapes,
        faceHeight: Float
    ) {
        if (landmarks.size < 468) return
        
        // 计算上下眼睑的距离
        val leftUpperLid = listOf(159, 158, 157, 173).map { landmarks[it] }
        val leftLowerLid = listOf(144, 145, 153, 154).map { landmarks[it] }
        
        val rightUpperLid = listOf(386, 385, 384, 398).map { landmarks[it] }
        val rightLowerLid = listOf(373, 374, 380, 381).map { landmarks[it] }
        
        // 计算眼睑距离
        fun calculateEyeOpenness(upper: List<Point3D>, lower: List<Point3D>): Float {
            var totalDist = 0f
            val count = minOf(upper.size, lower.size)
            for (i in 0 until count) {
                totalDist += upper[i].distance2DTo(lower[i])
            }
            return totalDist / count
        }
        
        val leftOpenness = calculateEyeOpenness(leftUpperLid, leftLowerLid) / faceHeight
        val rightOpenness = calculateEyeOpenness(rightUpperLid, rightLowerLid) / faceHeight
        
        // 正常睁眼的基准值
        val openBaseline = 0.035f
        val squintThreshold = 0.025f
        
        // 当眼睛开度减小但不是完全闭眼时，认为是 squint
        // 需要结合 eyeBlink 来判断（squint 时 blink 值较小）
        val leftSquintValue = if (leftOpenness < openBaseline && leftOpenness > squintThreshold) {
            (openBaseline - leftOpenness) / (openBaseline - squintThreshold)
        } else {
            0f
        }
        
        val rightSquintValue = if (rightOpenness < openBaseline && rightOpenness > squintThreshold) {
            (openBaseline - rightOpenness) / (openBaseline - squintThreshold)
        } else {
            0f
        }
        
        blendShapes.eyeSquintLeft = leftSquintValue.coerceIn(0f, 1f)
        blendShapes.eyeSquintRight = rightSquintValue.coerceIn(0f, 1f)
    }
}
