package com.example.virtualfacecapture

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.example.commondata.FaceLandmarkIndices
import com.example.commondata.FaceVisualizationData
import com.example.commondata.HeadPoseData
import com.example.commondata.Point3D
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicReference

/**
 * Custom View for rendering camera preview with face landmarks overlay
 * Uses double-buffering with AtomicReference to avoid race conditions
 */
class CameraPreviewView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    // Double-buffered bitmap using AtomicReference for thread-safe swap
    private val displayBitmap = AtomicReference<Bitmap?>(null)
    
    // Working bitmap (only accessed from update thread)
    private var workingBitmap: Bitmap? = null
    private var lastWidth = 0
    private var lastHeight = 0
    
    // Reusable pixel array
    private var pixelBuffer: IntArray? = null
    
    private val bitmapPaint = Paint().apply {
        isFilterBitmap = true
        isAntiAlias = true
    }
    
    private val detectionBoxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }
    
    // 普通点位（灰色）
    // 普通点位（白色，半透明）
    private val landmarkPaint = Paint().apply {
        color = Color.argb(180, 255, 255, 255)  // 半透明白色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // ============ 分部位颜色定义 ============
    
    // 图像右侧眼睛- 蓝色
    private val imageRightEyePaint = Paint().apply {
        color = Color.rgb(0, 150, 255)  // 浅蓝
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 图像左侧眼睛- 青色
    private val imageLeftEyePaint = Paint().apply {
        color = Color.rgb(0, 255, 200)  // 青色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 图像右侧眉毛 - 紫色
    private val imageRightEyebrowPaint = Paint().apply {
        color = Color.rgb(180, 100, 255)  // 紫色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 图像左侧眉毛 - 粉色
    private val imageLeftEyebrowPaint = Paint().apply {
        color = Color.rgb(255, 100, 180)  // 粉色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 鼻子 - 橙色
    private val nosePaint = Paint().apply {
        color = Color.rgb(255, 165, 0)  // 橙色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 嘴巴外轮廓 - 白色
    private val mouthOuterPaint = Paint().apply {
        color = Color.rgb(255, 255, 255)  // 白色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 嘴巴内轮廓 - 深红色
    private val mouthInnerPaint = Paint().apply {
        color = Color.rgb(200, 50, 50)  // 深红色
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val mouthInnerKeyPaint = Paint().apply {
        color = Color.rgb(0, 255, 0)  // 深红色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 脸部轮廓 - 绿色
    private val faceOvalPaint = Paint().apply {
        color = Color.rgb(100, 255, 100)  // 绿色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 虹膜中心 - 黄色
    private val irisCenterPaint = Paint().apply {
        color = Color.rgb(255, 255, 0)  // 黄色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // 虹膜边缘 - 金色
    private val irisEdgePaint = Paint().apply {
        color = Color.rgb(255, 200, 0)  // 金色
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val meshLinePaint = Paint().apply {
        color = Color.argb(100, 0, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 1f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 36f
        isAntiAlias = true
        typeface = Typeface.MONOSPACE
    }
    
    private val textBackgroundPaint = Paint().apply {
        color = Color.argb(150, 0, 0, 0)
        style = Paint.Style.FILL
    }
    
    private val srcRect = Rect()
    private val dstRect = RectF()
    
    // Thread-safe visualization data
    private val visualizationDataRef = AtomicReference<FaceVisualizationData?>(null)
    
    // Drawing options
    var drawLandmarks = true
    var drawDetectionBox = true
    var drawMesh = false
    var drawKeyPoints = true
    var drawIris = true
    var mirrorHorizontal = true
    var rotationDegrees = 0f
    
    companion object {
        // 使用统一的点位定义
        private val IMAGE_RIGHT_EYE_INDICES = FaceLandmarkIndices.ImageRightEye.ALL_EYELID.toSet()
        private val IMAGE_LEFT_EYE_INDICES = FaceLandmarkIndices.ImageLeftEye.ALL_EYELID.toSet()
        private val IMAGE_RIGHT_EYEBROW_INDICES = FaceLandmarkIndices.ImageRightEyebrow.ALL.toSet()
        private val IMAGE_LEFT_EYEBROW_INDICES = FaceLandmarkIndices.ImageLeftEyebrow.ALL.toSet()
        private val NOSE_INDICES = FaceLandmarkIndices.Nose.KEY_POINTS.toSet()
        // 使用完整的嘴巴点位定义进行绘制
        private val MOUTH_OUTER_INDICES = FaceLandmarkIndices.Mouth.OUTER_FULL.toSet()
        private val MOUTH_INNER_INDICES = FaceLandmarkIndices.Mouth.INNER_FULL.toSet()
        private val FACE_OVAL_INDICES = FaceLandmarkIndices.FaceOval.ALL.toSet()
        private val IRIS_CENTER_INDICES = FaceLandmarkIndices.Iris.ALL_CENTERS.toSet()
        private val IRIS_ALL_INDICES = FaceLandmarkIndices.Iris.ALL.toSet()
    }
    
    /**
     * Update camera frame - thread-safe with double buffering
     * Camera now outputs already-rotated frames, so no rotation needed here
     */
    fun updateCameraFrame(rgbBuffer: ByteBuffer, width: Int, height: Int) {
        val pixelCount = width * height
        
        // Reuse pixel buffer
        if (pixelBuffer == null || pixelBuffer!!.size != pixelCount) {
            pixelBuffer = IntArray(pixelCount)
        }
        val pixels = pixelBuffer!!
        
        // Convert RGB to ARGB
        rgbBuffer.rewind()
        for (i in pixels.indices) {
            val r = rgbBuffer.get().toInt() and 0xFF
            val g = rgbBuffer.get().toInt() and 0xFF
            val b = rgbBuffer.get().toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        rgbBuffer.rewind()
        
        // Create or reuse bitmap
        if (workingBitmap == null || lastWidth != width || lastHeight != height) {
            workingBitmap?.recycle()
            workingBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            lastWidth = width
            lastHeight = height
        }
        
        workingBitmap!!.setPixels(pixels, 0, width, 0, 0, width, height)
        
        // No rotation needed - camera already outputs correctly oriented frames
        // Just copy the bitmap for display
        val newDisplayBitmap = workingBitmap!!.copy(Bitmap.Config.ARGB_8888, false)
        
        // Atomic swap
        val oldBitmap = displayBitmap.getAndSet(newDisplayBitmap)
        
        oldBitmap?.let { old ->
            post {
                if (!old.isRecycled) {
                    old.recycle()
                }
            }
        }
        
        postInvalidate()
    }
    
    fun updateVisualizationData(data: FaceVisualizationData?) {
        visualizationDataRef.set(data)
        postInvalidate()
    }
    
    fun clear() {
        displayBitmap.getAndSet(null)?.recycle()
        workingBitmap?.recycle()
        workingBitmap = null
        visualizationDataRef.set(null)
        pixelBuffer = null
        lastWidth = 0
        lastHeight = 0
        postInvalidate()
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val bitmap = displayBitmap.get() ?: return
        if (bitmap.isRecycled) return
        
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val bitmapWidth = bitmap.width.toFloat()
        val bitmapHeight = bitmap.height.toFloat()
        
        val viewAspect = viewWidth / viewHeight
        val bitmapAspect = bitmapWidth / bitmapHeight
        
        val scale: Float
        val dx: Float
        val dy: Float
        
        if (bitmapAspect > viewAspect) {
            scale = viewWidth / bitmapWidth
            dx = 0f
            dy = (viewHeight - bitmapHeight * scale) / 2
        } else {
            scale = viewHeight / bitmapHeight
            dx = (viewWidth - bitmapWidth * scale) / 2
            dy = 0f
        }
        
        srcRect.set(0, 0, bitmap.width, bitmap.height)
        dstRect.set(dx, dy, dx + bitmapWidth * scale, dy + bitmapHeight * scale)
        
        try {
            if (mirrorHorizontal) {
                canvas.save()
                canvas.scale(-1f, 1f, viewWidth / 2, viewHeight / 2)
                canvas.drawBitmap(bitmap, srcRect, dstRect, bitmapPaint)
                canvas.restore()
            } else {
                canvas.drawBitmap(bitmap, srcRect, dstRect, bitmapPaint)
            }
        } catch (e: Exception) {
            return
        }
        
        visualizationDataRef.get()?.let { data ->
            drawFaceVisualization(canvas, data, dstRect)
            drawPoseInfo(canvas, data.headPose)
        }
    }
    
    private fun drawFaceVisualization(canvas: Canvas, data: FaceVisualizationData, dstRect: RectF) {
        val scaleX = dstRect.width()
        val scaleY = dstRect.height()
        val offsetX = dstRect.left
        val offsetY = dstRect.top
        
        fun transformPoint(lx: Float, ly: Float): Pair<Float, Float> {
            var tx = lx
            val ty = ly
            
            if (mirrorHorizontal) {
                tx = 1f - tx
            }
            
            return Pair(offsetX + tx * scaleX, offsetY + ty * scaleY)
        }
        
        // Detection box
        if (drawDetectionBox && data.detectionBox != null) {
            val box = data.detectionBox!!
            val (x1, y1) = transformPoint(box.x1, box.y1)
            val (x2, y2) = transformPoint(box.x2, box.y2)
            
            val boxRect = RectF(minOf(x1, x2), minOf(y1, y2), maxOf(x1, x2), maxOf(y1, y2))
            canvas.drawRect(boxRect, detectionBoxPaint)
            canvas.drawText(String.format("%.2f", box.score), boxRect.left + 5, boxRect.top - 5, textPaint)
        }
        
        // Landmarks - 按部位用不同颜色绘制
        if (drawLandmarks && data.landmarks.isNotEmpty()) {
            for ((index, landmark) in data.landmarks.withIndex()) {
                val (x, y) = transformPoint(landmark.x, landmark.y)
                
                // 根据点位所属部位选择颜色和大小
                val (paint, radius) = when {
                    // 虹膜（最高优先级）
                    drawIris && index in IRIS_CENTER_INDICES -> Pair(irisCenterPaint, 6f)
                    drawIris && index in IRIS_ALL_INDICES -> Pair(irisEdgePaint, 4f)
                    
                    // 眼睛
                    drawKeyPoints && index in IMAGE_RIGHT_EYE_INDICES -> Pair(imageRightEyePaint, 4f)
                    drawKeyPoints && index in IMAGE_LEFT_EYE_INDICES -> Pair(imageLeftEyePaint, 4f)
                    
                    // 眉毛
                    drawKeyPoints && index in IMAGE_RIGHT_EYEBROW_INDICES -> Pair(imageRightEyebrowPaint, 4f)
                    drawKeyPoints && index in IMAGE_LEFT_EYEBROW_INDICES -> Pair(imageLeftEyebrowPaint, 4f)
                    
                    // 鼻子
                    drawKeyPoints && index in NOSE_INDICES -> Pair(nosePaint, 4f)
                    
                    // 嘴巴
                    drawKeyPoints && index in MOUTH_OUTER_INDICES -> Pair(mouthOuterPaint, 4f)
                    drawKeyPoints && index in MOUTH_INNER_INDICES && drawKeyPoints && index == FaceLandmarkIndices.Mouth.INNER_FULL[0] -> Pair(mouthInnerKeyPaint, 3f)
                    drawKeyPoints && index in MOUTH_INNER_INDICES && drawKeyPoints && index == FaceLandmarkIndices.Mouth.INNER_FULL[10] -> Pair(mouthInnerKeyPaint, 3f)
                    drawKeyPoints && index in MOUTH_INNER_INDICES && drawKeyPoints && index == FaceLandmarkIndices.Mouth.INNER_FULL[5] -> Pair(mouthInnerKeyPaint, 3f)
                    drawKeyPoints && index in MOUTH_INNER_INDICES -> Pair(mouthInnerPaint, 3f)

                    // 脸部轮廓
                    drawKeyPoints && index in FACE_OVAL_INDICES -> Pair(faceOvalPaint, 3f)
                    
                    // 普通点
                    else -> Pair(landmarkPaint, 2f)
                }
                
                canvas.drawCircle(x, y, radius, paint)
            }
        }
        
        // Mesh
        if (drawMesh && data.landmarks.size >= 468) {
            // 眉毛连线
            drawLandmarkPath(canvas, data.landmarks, FaceLandmarkIndices.ImageRightEyebrow.ALL, ::transformPoint, imageRightEyebrowPaint)
            drawLandmarkPath(canvas, data.landmarks, FaceLandmarkIndices.ImageLeftEyebrow.ALL, ::transformPoint, imageLeftEyebrowPaint)
            
            // 脸部轮廓连线
            drawLandmarkPath(canvas, data.landmarks, FaceLandmarkIndices.FaceOval.ALL, ::transformPoint, meshLinePaint)
        }
    }
    
    /**
     * 绘制点位连线
     */
    private fun drawLandmarkPath(
        canvas: Canvas,
        landmarks: List<Point3D>,
        indices: List<Int>,
        transformPoint: (Float, Float) -> Pair<Float, Float>,
        paint: Paint
    ) {
        val path = Path()
        var first = true
        for (index in indices) {
            if (index >= landmarks.size) continue
            val (x, y) = transformPoint(landmarks[index].x, landmarks[index].y)
            if (first) {
                path.moveTo(x, y)
                first = false
            } else {
                path.lineTo(x, y)
            }
        }
        
        val linePaint = Paint(paint).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            alpha = 150
        }
        canvas.drawPath(path, linePaint)
    }
    
    private fun drawPoseInfo(canvas: Canvas, pose: HeadPoseData) {
        val padding = 16f
        val lineHeight = 40f
        val x = padding
        var y = height - padding - lineHeight * 3
        
        canvas.drawRoundRect(RectF(x - 8, y - lineHeight, x + 280, y + lineHeight * 3 + 8), 8f, 8f, textBackgroundPaint)
        
        canvas.drawText(String.format("Pitch: %+6.1f°", pose.pitch), x, y, textPaint)
        y += lineHeight
        canvas.drawText(String.format("Yaw:   %+6.1f°", pose.yaw), x, y, textPaint)
        y += lineHeight
        canvas.drawText(String.format("Roll:  %+6.1f°", pose.roll), x, y, textPaint)
    }
    
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        clear()
    }
}