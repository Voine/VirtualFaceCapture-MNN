package com.example.commondata

/**
 * MediaPipe 478 点位的统一索引定义
 * 
 * 所有面部关键点的索引都在这里定义，供整个项目使用。
 * 
 * MediaPipe 478 点位说明：
 * - 0-467: 标准面部网格点
 * - 468-472: 左虹膜 (468=中心, 469-472=边缘：上、右、下、左)
 * - 473-477: 右虹膜 (473=中心, 474-477=边缘：上、右、下、左)
 * 
 * 坐标系说明（前置摄像头镜像模式）：
 * - 图像左侧 = 真人右侧
 * - 图像右侧 = 真人左侧
 * - 为避免混淆，这里使用 IMAGE_LEFT / IMAGE_RIGHT 命名
 */
object FaceLandmarkIndices {
    
    // ==================== 眼睛 ====================
    
    /**
     * 图像右侧眼睛
     * 
     * 官方连接路径：
     * - 下眼睑: 33 → 7 → 163 → 144 → 145 → 153 → 154 → 155 → 133
     * - 上眼睑: 33 → 246 → 161 → 160 → 159 → 158 → 157 → 173 → 133
     */
    object ImageRightEye {
        val UPPER_EYELID = listOf(246, 161, 160, 159, 158, 157, 173)  // 上眼睑 7 个点
        val LOWER_EYELID = listOf(7, 163, 144, 145, 153, 154, 155)    // 下眼睑 7 个点
        const val INNER_CORNER = 133   // 内眼角（靠近鼻子）
        const val OUTER_CORNER = 33    // 外眼角
        const val IRIS_CENTER = 468    // 虹膜中心
        val IRIS_EDGE = listOf(469, 470, 471, 472)  // 虹膜边缘（上、右、下、左）
        
        val ALL_EYELID = UPPER_EYELID + LOWER_EYELID + listOf(INNER_CORNER, OUTER_CORNER)
        val ALL_IRIS = listOf(IRIS_CENTER) + IRIS_EDGE
    }
    
    /**
     * 图像左侧眼睛
     * 
     * 官方连接路径：
     * - 下眼睑: 263 → 249 → 390 → 373 → 374 → 380 → 381 → 382 → 362
     * - 上眼睑: 263 → 466 → 388 → 387 → 386 → 385 → 384 → 398 → 362
     */
    object ImageLeftEye {
        val UPPER_EYELID = listOf(466, 388, 387, 386, 385, 384, 398)   // 上眼睑 7 个点
        val LOWER_EYELID = listOf(249, 390, 373, 374, 380, 381, 382)   // 下眼睑 7 个点
        const val INNER_CORNER = 362    // 内眼角（靠近鼻子）
        const val OUTER_CORNER = 263    // 外眼角
        const val IRIS_CENTER = 473     // 虹膜中心
        val IRIS_EDGE = listOf(474, 475, 476, 477)  // 虹膜边缘（上、右、下、左）
        
        val ALL_EYELID = UPPER_EYELID + LOWER_EYELID + listOf(INNER_CORNER, OUTER_CORNER)
        val ALL_IRIS = listOf(IRIS_CENTER) + IRIS_EDGE
    }
    
    // ==================== 眉毛 ====================
    
    /**
     * 图像右侧眉毛
     * 
     * 官方连接路径：
     * - 46 → 53 → 52 → 65 → 55 (上弧)
     * - 70 → 63 → 105 → 66 → 107 (下弧)
     */
    object ImageRightEyebrow {
        const val INNER = 107     // 眉头（靠近鼻子）
        const val MIDDLE = 105    // 眉中
        const val OUTER = 46      // 眉尾
        val UPPER_ARC = listOf(46, 53, 52, 65, 55)
        val LOWER_ARC = listOf(70, 63, 105, 66, 107)
        val ALL = listOf(107, 66, 105, 63, 70, 46, 53, 52, 65, 55)
    }
    
    /**
     * 图像左侧眉毛
     * 
     * 官方连接路径：
     * - 276 → 283 → 282 → 295 → 285 (上弧)
     * - 300 → 293 → 334 → 296 → 336 (下弧)
     */
    object ImageLeftEyebrow {
        const val INNER = 336     // 眉头（靠近鼻子）
        const val MIDDLE = 334    // 眉中
        const val OUTER = 276     // 眉尾
        val UPPER_ARC = listOf(276, 283, 282, 295, 285)
        val LOWER_ARC = listOf(300, 293, 334, 296, 336)
        val ALL = listOf(336, 296, 334, 293, 300, 276, 283, 282, 295, 285)
    }
    
    // ==================== 鼻子 ====================
    
    object Nose {
        const val TIP = 1             // 鼻尖
        const val ROOT = 168          // 鼻根（两眼之间）
        const val IMAGE_LEFT_WING = 98      // 左鼻翼（图像左侧）
        const val IMAGE_RIGHT_WING = 327    // 右鼻翼（图像右侧）
        const val LEFT_NOSTRIL = 219  // 左鼻孔
        const val RIGHT_NOSTRIL = 439 // 右鼻孔
        
        val BRIDGE = listOf(6, 197, 195, 5, 4)  // 鼻梁（从上到下）
        
        // 鼻子皱纹区域（用于 noseSneer）
        val LEFT_SNEER_AREA = listOf(49, 129, 102, 48)
        val RIGHT_SNEER_AREA = listOf(279, 358, 331, 278)
        
        val KEY_POINTS = listOf(TIP, ROOT, IMAGE_LEFT_WING, IMAGE_RIGHT_WING) + BRIDGE
    }
    
    // ==================== 嘴巴 ====================
    
    object Mouth {
        // 嘴角（外轮廓端点）
        const val IMAGE_LEFT_CORNER = 61        // 左嘴角（图像左侧，外轮廓）
        const val IMAGE_RIGHT_CORNER = 291      // 右嘴角（图像右侧，外轮廓）
        
        // 嘴角（内轮廓端点，用于 smile 计算，水平对齐）
        const val IMAGE_LEFT_CORNER_INNER = 78  // 左嘴角（内轮廓）
        const val IMAGE_RIGHT_CORNER_INNER = 308 // 右嘴角（内轮廓）
        
        // 上唇
        const val UPPER_LIP_TOP = 0       // 上唇最上点（外轮廓中心）
        const val UPPER_LIP_CENTER = 13   // 上唇中心（内轮廓）
        val UPPER_LIP_IMAGE_LEFT = listOf(40, 39, 37)
        val UPPER_LIP_IMAGE_RIGHT = listOf(270, 269, 267)
        
        // 下唇
        const val LOWER_LIP_BOTTOM = 17   // 下唇最下点（外轮廓中心）
        const val LOWER_LIP_CENTER = 14   // 下唇中心（内轮廓）
        val LOWER_LIP_IMAGE_LEFT = listOf(84, 181, 91)
        val LOWER_LIP_IMAGE_RIGHT = listOf(314, 405, 321)
        
        // 唇内侧（用于 mouthRoll）
        const val UPPER_LIP_INNER = 12
        const val LOWER_LIP_INNER = 15
        
        // 酒窝区域
        const val IMAGE_LEFT_DIMPLE = 206
        const val IMAGE_RIGHT_DIMPLE = 426

        /**
         * 嘴巴外轮廓 - 完整 20 点版本（用于绘制）
         * 
         * 官方路径：
         * - 上半部分 (左→右): 61 → 146 → 91 → 181 → 84 → 17 → 314 → 405 → 321 → 375 → 291
         * - 下半部分 (右→左): 291 → 409 → 270 → 269 → 267 → 0 → 37 → 39 → 40 → 185 → 61
         */
        val OUTER_FULL = listOf(
            61,   // [0] 左嘴角 ⭐
            146,  // [1]
            91,   // [2]
            181,  // [3]
            84,   // [4]
            17,   // [5] 下唇外轮廓中心 ⭐
            314,  // [6]
            405,  // [7]
            321,  // [8]
            375,  // [9]
            291,  // [10] 右嘴角 ⭐
            409,  // [11]
            270,  // [12]
            269,  // [13]
            267,  // [14]
            0,    // [15] 上唇外轮廓中心 ⭐
            37,   // [16]
            39,   // [17]
            40,   // [18]
            185   // [19]
        )
        
        /**
         * 嘴巴内轮廓 - 完整 20 点版本（用于绘制）
         * 
         * 官方路径：
         * - 上半部分 (左→右): 78 → 191 → 80 → 81 → 82 → 13 → 312 → 311 → 310 → 415 → 308
         * - 下半部分 (右→左): 308 → 324 → 318 → 402 → 317 → 14 → 87 → 178 → 88 → 95 → 78
         */
        val INNER_FULL = listOf(
            78,   // [0] 左嘴角（内轮廓）⭐
            191,  // [1]
            80,   // [2]
            81,   // [3]
            82,   // [4]
            13,   // [5] 上唇内轮廓中心 ⭐
            312,  // [6]
            311,  // [7]
            310,  // [8]
            415,  // [9]
            308,  // [10] 右嘴角（内轮廓）⭐
            324,  // [11]
            318,  // [12]
            402,  // [13]
            317,  // [14]
            14,   // [15] 下唇内轮廓中心 ⭐
            87,   // [16]
            178,  // [17]
            88,   // [18]
            95    // [19]
        )
        
        val KEY_POINTS = listOf(IMAGE_LEFT_CORNER, IMAGE_RIGHT_CORNER, UPPER_LIP_TOP, LOWER_LIP_BOTTOM,
                                IMAGE_LEFT_CORNER_INNER, IMAGE_RIGHT_CORNER_INNER, UPPER_LIP_CENTER, LOWER_LIP_CENTER)
    }
    
    // ==================== 脸颊 ====================
    
    object Cheek {
        // 脸颊鼓起区域
        val IMAGE_LEFT = listOf(36, 205, 206, 187, 123)
        val IMAGE_RIGHT = listOf(266, 425, 426, 411, 352)
        
        // 脸颊挤压区域（靠近眼睛下方）
        const val IMAGE_LEFT_SQUINT = 117
        const val IMAGE_RIGHT_SQUINT = 346
        const val IMAGE_LEFT_LOWER_EYELID = 111
        const val IMAGE_RIGHT_LOWER_EYELID = 340
    }
    
    // ==================== 下巴 ====================
    
    object Jaw {
        const val CHIN_CENTER = 152   // 下巴中心
        const val CHIN_IMAGE_LEFT = 172     // 下巴左侧
        const val CHIN_IMAGE_RIGHT = 397    // 下巴右侧
        const val IMAGE_LEFT_JAW = 132      // 左下颌
        const val IMAGE_RIGHT_JAW = 361     // 右下颌
    }
    
    // ==================== 脸部轮廓 ====================
    
    object FaceOval {
        /**
         * 脸部轮廓（36 个点，顺时针从额头顶部开始）
         */
        val ALL = listOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132,
            93, 234, 127, 162, 21, 54, 103, 67, 109
        )
        
        const val TOP = 10            // 额头顶部
        const val BOTTOM = 152        // 下巴底部
        const val IMAGE_LEFT_TEMPLE = 454   // 左太阳穴
        const val IMAGE_RIGHT_TEMPLE = 234  // 右太阳穴
    }
    
    // ==================== 虹膜 ====================
    
    object Iris {
        // 图像右侧虹膜
        const val IMAGE_RIGHT_CENTER = 468
        val IMAGE_RIGHT_EDGE = listOf(469, 470, 471, 472)
        val IMAGE_RIGHT_ALL = listOf(468, 469, 470, 471, 472)
        
        // 图像左侧虹膜
        const val IMAGE_LEFT_CENTER = 473
        val IMAGE_LEFT_EDGE = listOf(474, 475, 476, 477)
        val IMAGE_LEFT_ALL = listOf(473, 474, 475, 476, 477)
        
        val ALL_CENTERS = listOf(IMAGE_RIGHT_CENTER, IMAGE_LEFT_CENTER)
        val ALL = IMAGE_RIGHT_ALL + IMAGE_LEFT_ALL
    }
    /**
     * 从 MediaPipe 468 点中提取关键点
     *
     * @param landmarks MediaPipe 输出的 468 个点 (像素坐标或归一化坐标)
     * @param imageWidth 图像宽度 (如果是归一化坐标，用于转换)
     * @param imageHeight 图像高度 (如果是归一化坐标，用于转换)
     * @return 提取的关键点集合
     */
    fun extractKeyPoints(
        landmarks: List<Point3D>,
        imageWidth: Int = 0,
        imageHeight: Int = 0
    ): FaceKeyPoints {
        // 如果提供了图像尺寸且坐标是归一化的，则转换为像素坐标
        val points = if (imageWidth > 0 && imageHeight > 0 &&
            landmarks[0].x <= 1.0f && landmarks[0].y <= 1.0f) {
            landmarks.map {
                Point3D(
                    it.x * imageWidth,
                    it.y * imageHeight,
                    it.z
                )
            }
        } else {
            landmarks
        }

        // 提取眼部点
        val imageRightEye = EyePoints(
            upper = ImageRightEye.UPPER_EYELID.map { points[it] },
            lower = ImageRightEye.LOWER_EYELID.map { points[it] },
            innerCorner = points[ImageRightEye.INNER_CORNER],
            outerCorner = points[ImageRightEye.OUTER_CORNER],
            irisCenter = if (points.size > ImageRightEye.IRIS_CENTER)
                points[ImageRightEye.IRIS_CENTER] else null
        )

        val imageLeftEye = EyePoints(
            upper = ImageLeftEye.UPPER_EYELID.map { points[it] },
            lower = ImageLeftEye.LOWER_EYELID.map { points[it] },
            innerCorner = points[ImageLeftEye.INNER_CORNER],
            outerCorner = points[ImageLeftEye.OUTER_CORNER],
            irisCenter = if (points.size > ImageLeftEye.IRIS_CENTER)
                points[ImageLeftEye.IRIS_CENTER] else null
        )

        // 提取眉毛点
        val imageLeftEyebrow = ImageLeftEyebrow.UPPER_ARC.map { points[it] }
        val imageRightEyebrow = ImageRightEyebrow.UPPER_ARC.map { points[it] }

        // 提取鼻子点
        val nose = listOf(
            points[Nose.ROOT],
            points[Nose.TIP],
            points[Nose.IMAGE_LEFT_WING],
            points[Nose.IMAGE_RIGHT_WING]
        ) + Nose.BRIDGE.map { points[it] }

        // 提取嘴巴点
        val mouthOuter = Mouth.OUTER_FULL.map { points[it] }
        val mouthInner = Mouth.INNER_FULL.map { points[it] }

        // 提取轮廓点
        val contour = FaceOval.ALL.map { points[it] }

        // 计算面部尺寸参考
        val faceWidth = imageRightEye.outerCorner.distance2DTo(imageLeftEye.outerCorner)
        val faceHeight = points[Nose.ROOT].distance2DTo(points[FaceOval.BOTTOM])

        return FaceKeyPoints(
            imageLeftEye = imageLeftEye,
            imageRightEye = imageRightEye,
            imageLeftEyebrow = imageLeftEyebrow,
            imageRightEyebrow = imageRightEyebrow,
            nose = nose,
            mouthOuter = mouthOuter,
            mouthInner = mouthInner,
            contour = contour,
            faceWidth = faceWidth,
            faceHeight = faceHeight
        )
    }
}

/**
 * 头部姿态数据类
 *
 * 坐标系定义（与 UI 显示一致）：
 * - pitch < 0 = 真人抬头 (皮套抬头)
 * - pitch > 0 = 真人低头 (皮套低头)
 * - yaw < 0 = 真人向右转 (皮套向左转)
 * - yaw > 0 = 真人向左转 (皮套向右转)
 */
data class HeadPose(
    val pitch: Float = 0f,  // 俯仰角 (抬头负, 低头正)
    val yaw: Float = 0f,    // 偏航角 (右转负, 左转正)
    val roll: Float = 0f    // 翻滚角
)

const val FRAME_COUNT_CALIBRATE = 20
