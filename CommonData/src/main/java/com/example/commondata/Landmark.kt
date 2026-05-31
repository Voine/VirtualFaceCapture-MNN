/**
 * MediaPipe Landmark 数据结构
 * 基于 OpenSeeFace 工程实践的 Kotlin 实现
 */

package com.example.commondata

import kotlin.math.sqrt
import kotlin.math.pow

/**
 * 3D 点坐标
 * @param x 归一化的 X 坐标 [0, 1] 或像素坐标
 * @param y 归一化的 Y 坐标 [0, 1] 或像素坐标
 * @param z 深度坐标 (MediaPipe 的相对深度)
 */
data class Point3D(
    val x: Float,
    val y: Float,
    val z: Float
) {
    /**
     * 计算到另一个点的欧氏距离
     */
    fun distanceTo(other: Point3D): Float {
        return sqrt(
            (x - other.x).pow(2) +
            (y - other.y).pow(2) +
            (z - other.z).pow(2)
        )
    }

    /**
     * 计算 2D 距离 (仅考虑 XY)
     */
    fun distance2DTo(other: Point3D): Float {
        return sqrt(
            (x - other.x).pow(2) +
            (y - other.y).pow(2)
        )
    }

    /**
     * 向量减法
     */
    operator fun minus(other: Point3D): Point3D {
        return Point3D(x - other.x, y - other.y, z - other.z)
    }

    /**
     * 向量加法
     */
    operator fun plus(other: Point3D): Point3D {
        return Point3D(x + other.x, y + other.y, z + other.z)
    }

    /**
     * 标量乘法
     */
    operator fun times(scalar: Float): Point3D {
        return Point3D(x * scalar, y * scalar, z * scalar)
    }

    /**
     * 标量除法
     */
    operator fun div(scalar: Float): Point3D {
        return Point3D(x / scalar, y / scalar, z / scalar)
    }
}

/**
 * 面部关键点集合
 * 包含从 MediaPipe 468 点中提取的关键特征点
 */
data class FaceKeyPoints(
    // 眼部区域
    val imageLeftEye: EyePoints,
    val imageRightEye: EyePoints,

    // 眉毛
    val imageLeftEyebrow: List<Point3D>,
    val imageRightEyebrow: List<Point3D>,

    // 鼻子
    val nose: List<Point3D>,

    // 嘴巴
    val mouthOuter: List<Point3D>,
    val mouthInner: List<Point3D>,

    // 轮廓
    val contour: List<Point3D>,

    // 面部尺寸参考点
    val faceWidth: Float,  // 眼角间距
    val faceHeight: Float  // 鼻根到下巴距离
)

/**
 * 眼部关键点
 */
data class EyePoints(
    val upper: List<Point3D>,      // 上眼睑
    val lower: List<Point3D>,      // 下眼睑
    val innerCorner: Point3D,      // 内眼角
    val outerCorner: Point3D,      // 外眼角
    val irisCenter: Point3D? = null // 虹膜中心 (MediaPipe 提供)
)

/**
 * ARKit BlendShape 输出
 * 包含 52 个标准 BlendShape 参数
 */
data class ARKitBlendShapes(
    // 眼部 BlendShape (12个)
    var eyeBlinkLeft: Float = 0f,
    var eyeBlinkRight: Float = 0f,
    var eyeWideLeft: Float = 0f,
    var eyeWideRight: Float = 0f,
    var eyeSquintLeft: Float = 0f,
    var eyeSquintRight: Float = 0f,
    var eyeLookUpLeft: Float = 0f,
    var eyeLookUpRight: Float = 0f,
    var eyeLookDownLeft: Float = 0f,
    var eyeLookDownRight: Float = 0f,
    var eyeLookInLeft: Float = 0f,
    var eyeLookOutLeft: Float = 0f,
    var eyeLookInRight: Float = 0f,
    var eyeLookOutRight: Float = 0f,

    // 眉毛 BlendShape (8个)
    var browDownLeft: Float = 0f,
    var browDownRight: Float = 0f,
    var browInnerUp: Float = 0f,
    var browOuterUpLeft: Float = 0f,
    var browOuterUpRight: Float = 0f,

    // 脸颊 BlendShape (4个)
    var cheekPuff: Float = 0f,
    var cheekSquintLeft: Float = 0f,
    var cheekSquintRight: Float = 0f,

    // 下颌 BlendShape (6个)
    var jawOpen: Float = 0f,
    var jawForward: Float = 0f,
    var jawLeft: Float = 0f,
    var jawRight: Float = 0f,

    // 嘴部 BlendShape (22个)
    var mouthClose: Float = 0f,
    var mouthFunnel: Float = 0f,
    var mouthPucker: Float = 0f,
    var mouthLeft: Float = 0f,
    var mouthRight: Float = 0f,
    var mouthSmileLeft: Float = 0f,
    var mouthSmileRight: Float = 0f,
    var mouthFrownLeft: Float = 0f,
    var mouthFrownRight: Float = 0f,
    var mouthDimpleLeft: Float = 0f,
    var mouthDimpleRight: Float = 0f,
    var mouthStretchLeft: Float = 0f,
    var mouthStretchRight: Float = 0f,
    var mouthRollLower: Float = 0f,
    var mouthRollUpper: Float = 0f,
    var mouthShrugLower: Float = 0f,
    var mouthShrugUpper: Float = 0f,
    var mouthPressLeft: Float = 0f,
    var mouthPressRight: Float = 0f,
    var mouthLowerDownLeft: Float = 0f,
    var mouthLowerDownRight: Float = 0f,
    var mouthUpperUpLeft: Float = 0f,
    var mouthUpperUpRight: Float = 0f,

    // 鼻子 BlendShape (2个)
    var noseSneerLeft: Float = 0f,
    var noseSneerRight: Float = 0f,

    // 舌头 (1个) - MediaPipe 不输出此项，但 ARKit 标准有
    var tongueOut: Float = 0f
) {
    companion object {
        // MediaPipe BlendShape 索引常量 (与 C++ BlendShapeIndex 枚举对应)
        const val BS_NEUTRAL = 0
        const val BS_BROW_DOWN_LEFT = 1
        const val BS_BROW_DOWN_RIGHT = 2
        const val BS_BROW_INNER_UP = 3
        const val BS_BROW_OUTER_UP_LEFT = 4
        const val BS_BROW_OUTER_UP_RIGHT = 5
        const val BS_CHEEK_PUFF = 6
        const val BS_CHEEK_SQUINT_LEFT = 7
        const val BS_CHEEK_SQUINT_RIGHT = 8
        const val BS_EYE_BLINK_LEFT = 9
        const val BS_EYE_BLINK_RIGHT = 10
        const val BS_EYE_LOOK_DOWN_LEFT = 11
        const val BS_EYE_LOOK_DOWN_RIGHT = 12
        const val BS_EYE_LOOK_IN_LEFT = 13
        const val BS_EYE_LOOK_IN_RIGHT = 14
        const val BS_EYE_LOOK_OUT_LEFT = 15
        const val BS_EYE_LOOK_OUT_RIGHT = 16
        const val BS_EYE_LOOK_UP_LEFT = 17
        const val BS_EYE_LOOK_UP_RIGHT = 18
        const val BS_EYE_SQUINT_LEFT = 19
        const val BS_EYE_SQUINT_RIGHT = 20
        const val BS_EYE_WIDE_LEFT = 21
        const val BS_EYE_WIDE_RIGHT = 22
        const val BS_JAW_FORWARD = 23
        const val BS_JAW_LEFT = 24
        const val BS_JAW_OPEN = 25
        const val BS_JAW_RIGHT = 26
        const val BS_MOUTH_CLOSE = 27
        const val BS_MOUTH_DIMPLE_LEFT = 28
        const val BS_MOUTH_DIMPLE_RIGHT = 29
        const val BS_MOUTH_FROWN_LEFT = 30
        const val BS_MOUTH_FROWN_RIGHT = 31
        const val BS_MOUTH_FUNNEL = 32
        const val BS_MOUTH_LEFT = 33
        const val BS_MOUTH_LOWER_DOWN_LEFT = 34
        const val BS_MOUTH_LOWER_DOWN_RIGHT = 35
        const val BS_MOUTH_PRESS_LEFT = 36
        const val BS_MOUTH_PRESS_RIGHT = 37
        const val BS_MOUTH_PUCKER = 38
        const val BS_MOUTH_RIGHT = 39
        const val BS_MOUTH_ROLL_LOWER = 40
        const val BS_MOUTH_ROLL_UPPER = 41
        const val BS_MOUTH_SHRUG_LOWER = 42
        const val BS_MOUTH_SHRUG_UPPER = 43
        const val BS_MOUTH_SMILE_LEFT = 44
        const val BS_MOUTH_SMILE_RIGHT = 45
        const val BS_MOUTH_STRETCH_LEFT = 46
        const val BS_MOUTH_STRETCH_RIGHT = 47
        const val BS_MOUTH_UPPER_UP_LEFT = 48
        const val BS_MOUTH_UPPER_UP_RIGHT = 49
        const val BS_NOSE_SNEER_LEFT = 50
        const val BS_NOSE_SNEER_RIGHT = 51
        const val BS_COUNT = 52
        
        /**
         * 从 MediaPipe BlendShape FloatArray 创建 ARKitBlendShapes
         * MediaPipe 输出的是按索引排列的 52 个值的数组
         * @param values MediaPipe 输出的 BlendShape 数组 (长度 52)
         * @return ARKitBlendShapes 对象
         */
        fun fromMediaPipeArray(values: FloatArray): ARKitBlendShapes {
            require(values.size >= BS_COUNT) { 
                "BlendShape array must have at least $BS_COUNT elements, got ${values.size}" 
            }
            return ARKitBlendShapes(
                // 眼部
                eyeBlinkLeft = values[BS_EYE_BLINK_LEFT],
                eyeBlinkRight = values[BS_EYE_BLINK_RIGHT],
                eyeWideLeft = values[BS_EYE_WIDE_LEFT],
                eyeWideRight = values[BS_EYE_WIDE_RIGHT],
                eyeSquintLeft = values[BS_EYE_SQUINT_LEFT],
                eyeSquintRight = values[BS_EYE_SQUINT_RIGHT],
                eyeLookUpLeft = values[BS_EYE_LOOK_UP_LEFT],
                eyeLookUpRight = values[BS_EYE_LOOK_UP_RIGHT],
                eyeLookDownLeft = values[BS_EYE_LOOK_DOWN_LEFT],
                eyeLookDownRight = values[BS_EYE_LOOK_DOWN_RIGHT],
                eyeLookInLeft = values[BS_EYE_LOOK_IN_LEFT],
                eyeLookOutLeft = values[BS_EYE_LOOK_OUT_LEFT],
                eyeLookInRight = values[BS_EYE_LOOK_IN_RIGHT],
                eyeLookOutRight = values[BS_EYE_LOOK_OUT_RIGHT],
                // 眉毛
                browDownLeft = values[BS_BROW_DOWN_LEFT],
                browDownRight = values[BS_BROW_DOWN_RIGHT],
                browInnerUp = values[BS_BROW_INNER_UP],
                browOuterUpLeft = values[BS_BROW_OUTER_UP_LEFT],
                browOuterUpRight = values[BS_BROW_OUTER_UP_RIGHT],
                // 脸颊
                cheekPuff = values[BS_CHEEK_PUFF],
                cheekSquintLeft = values[BS_CHEEK_SQUINT_LEFT],
                cheekSquintRight = values[BS_CHEEK_SQUINT_RIGHT],
                // 下颌
                jawOpen = values[BS_JAW_OPEN],
                jawForward = values[BS_JAW_FORWARD],
                jawLeft = values[BS_JAW_LEFT],
                jawRight = values[BS_JAW_RIGHT],
                // 嘴部
                mouthClose = values[BS_MOUTH_CLOSE],
                mouthFunnel = values[BS_MOUTH_FUNNEL],
                mouthPucker = values[BS_MOUTH_PUCKER],
                mouthLeft = values[BS_MOUTH_LEFT],
                mouthRight = values[BS_MOUTH_RIGHT],
                mouthSmileLeft = values[BS_MOUTH_SMILE_LEFT],
                mouthSmileRight = values[BS_MOUTH_SMILE_RIGHT],
                mouthFrownLeft = values[BS_MOUTH_FROWN_LEFT],
                mouthFrownRight = values[BS_MOUTH_FROWN_RIGHT],
                mouthDimpleLeft = values[BS_MOUTH_DIMPLE_LEFT],
                mouthDimpleRight = values[BS_MOUTH_DIMPLE_RIGHT],
                mouthStretchLeft = values[BS_MOUTH_STRETCH_LEFT],
                mouthStretchRight = values[BS_MOUTH_STRETCH_RIGHT],
                mouthRollLower = values[BS_MOUTH_ROLL_LOWER],
                mouthRollUpper = values[BS_MOUTH_ROLL_UPPER],
                mouthShrugLower = values[BS_MOUTH_SHRUG_LOWER],
                mouthShrugUpper = values[BS_MOUTH_SHRUG_UPPER],
                mouthPressLeft = values[BS_MOUTH_PRESS_LEFT],
                mouthPressRight = values[BS_MOUTH_PRESS_RIGHT],
                mouthLowerDownLeft = values[BS_MOUTH_LOWER_DOWN_LEFT],
                mouthLowerDownRight = values[BS_MOUTH_LOWER_DOWN_RIGHT],
                mouthUpperUpLeft = values[BS_MOUTH_UPPER_UP_LEFT],
                mouthUpperUpRight = values[BS_MOUTH_UPPER_UP_RIGHT],
                // 鼻子
                noseSneerLeft = values[BS_NOSE_SNEER_LEFT],
                noseSneerRight = values[BS_NOSE_SNEER_RIGHT],
                // 舌头 (MediaPipe 不输出)
                tongueOut = 0f
            )
        }
    }
    
    /**
     * 转换为 Map 格式 (方便序列化)
     */
    fun toMap(): Map<String, Float> {
        return mapOf(
            "eyeBlinkLeft" to eyeBlinkLeft,
            "eyeBlinkRight" to eyeBlinkRight,
            "eyeWideLeft" to eyeWideLeft,
            "eyeWideRight" to eyeWideRight,
            "eyeSquintLeft" to eyeSquintLeft,
            "eyeSquintRight" to eyeSquintRight,
            "eyeLookUpLeft" to eyeLookUpLeft,
            "eyeLookUpRight" to eyeLookUpRight,
            "eyeLookDownLeft" to eyeLookDownLeft,
            "eyeLookDownRight" to eyeLookDownRight,
            "eyeLookInLeft" to eyeLookInLeft,
            "eyeLookOutLeft" to eyeLookOutLeft,
            "eyeLookInRight" to eyeLookInRight,
            "eyeLookOutRight" to eyeLookOutRight,
            "browDownLeft" to browDownLeft,
            "browDownRight" to browDownRight,
            "browInnerUp" to browInnerUp,
            "browOuterUpLeft" to browOuterUpLeft,
            "browOuterUpRight" to browOuterUpRight,
            "cheekPuff" to cheekPuff,
            "cheekSquintLeft" to cheekSquintLeft,
            "cheekSquintRight" to cheekSquintRight,
            "jawOpen" to jawOpen,
            "jawForward" to jawForward,
            "jawLeft" to jawLeft,
            "jawRight" to jawRight,
            "mouthClose" to mouthClose,
            "mouthFunnel" to mouthFunnel,
            "mouthPucker" to mouthPucker,
            "mouthLeft" to mouthLeft,
            "mouthRight" to mouthRight,
            "mouthSmileLeft" to mouthSmileLeft,
            "mouthSmileRight" to mouthSmileRight,
            "mouthFrownLeft" to mouthFrownLeft,
            "mouthFrownRight" to mouthFrownRight,
            "mouthDimpleLeft" to mouthDimpleLeft,
            "mouthDimpleRight" to mouthDimpleRight,
            "mouthStretchLeft" to mouthStretchLeft,
            "mouthStretchRight" to mouthStretchRight,
            "mouthRollLower" to mouthRollLower,
            "mouthRollUpper" to mouthRollUpper,
            "mouthShrugLower" to mouthShrugLower,
            "mouthShrugUpper" to mouthShrugUpper,
            "mouthPressLeft" to mouthPressLeft,
            "mouthPressRight" to mouthPressRight,
            "mouthLowerDownLeft" to mouthLowerDownLeft,
            "mouthLowerDownRight" to mouthLowerDownRight,
            "mouthUpperUpLeft" to mouthUpperUpLeft,
            "mouthUpperUpRight" to mouthUpperUpRight,
            "noseSneerLeft" to noseSneerLeft,
            "noseSneerRight" to noseSneerRight,
            "tongueOut" to tongueOut
        )
    }

    /**
     * 限制所有值在 [0, 1] 范围内
     */
    fun clamp(): ARKitBlendShapes {
        return this.copy(
            eyeBlinkLeft = eyeBlinkLeft.coerceIn(0f, 1f),
            eyeBlinkRight = eyeBlinkRight.coerceIn(0f, 1f),
            eyeWideLeft = eyeWideLeft.coerceIn(0f, 1f),
            eyeWideRight = eyeWideRight.coerceIn(0f, 1f),
            // ... 其他字段同理
            jawOpen = jawOpen.coerceIn(0f, 1f),
            mouthSmileLeft = mouthSmileLeft.coerceIn(0f, 1f),
            mouthSmileRight = mouthSmileRight.coerceIn(0f, 1f)
        )
    }
}

