package com.example.virtualfacecapture

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

/**
 * 校准进度覆盖层 View
 * 
 * 在校准期间显示一个旋转的进度条和提示文字。
 * 设计为独立组件，可以在传统 View 系统或 Compose 中使用。
 * 
 * 使用方式：
 * 1. 传统 View：直接 addView 或在 XML 中使用
 * 2. Compose：使用 CalibrationOverlay() Composable
 * 
 * @author yrzhu
 * @date 2025/1/14
 */
class CalibrationOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    
    // ============ 配置参数 ============
    
    /**
     * 提示文字
     */
    var calibrationText: String = "面捕正在校准中..."
        set(value) {
            field = value
            invalidate()
        }
    
    /**
     * 进度条颜色
     */
    var progressColor: Int = Color.parseColor("#4CAF50")  // Material Green
        set(value) {
            field = value
            progressPaint.color = value
            invalidate()
        }
    
    /**
     * 背景遮罩颜色
     */
    var overlayColor: Int = Color.parseColor("#80000000")  // 50% 透明黑
        set(value) {
            field = value
            overlayPaint.color = value
            invalidate()
        }
    
    /**
     * 文字颜色
     */
    var textColor: Int = Color.WHITE
        set(value) {
            field = value
            textPaint.color = value
            invalidate()
        }
    
    /**
     * 进度条轨道颜色
     */
    var trackColor: Int = Color.parseColor("#40FFFFFF")  // 25% 透明白
        set(value) {
            field = value
            trackPaint.color = value
            invalidate()
        }
    
    /**
     * 进度条半径 (dp)
     */
    var progressRadius: Float = 40f
        set(value) {
            field = value
            updateDimensions()
            invalidate()
        }
    
    /**
     * 进度条宽度 (dp)
     */
    var progressStrokeWidth: Float = 4f
        set(value) {
            field = value
            updateDimensions()
            invalidate()
        }
    
    /**
     * 文字大小 (sp)
     */
    var textSize: Float = 16f
        set(value) {
            field = value
            updateDimensions()
            invalidate()
        }
    
    /**
     * 旋转一圈的时间 (ms)
     */
    var rotationDuration: Long = 1200L
    
    /**
     * 进度弧的角度范围
     */
    var sweepAngle: Float = 90f
    
    // ============ 内部状态 ============
    
    private var currentRotation: Float = 0f
    private var rotationAnimator: ValueAnimator? = null
    private var isAnimating: Boolean = false
    
    // 像素值缓存
    private var progressRadiusPx: Float = 0f
    private var progressStrokeWidthPx: Float = 0f
    private var textSizePx: Float = 0f
    
    // 绘制区域
    private val progressRect = RectF()
    
    // ============ 画笔 ============
    
    private val overlayPaint = Paint().apply {
        color = overlayColor
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val trackPaint = Paint().apply {
        color = trackColor
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    
    private val progressPaint = Paint().apply {
        color = progressColor
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = textColor
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }
    
    init {
        // 初始化时更新尺寸
        updateDimensions()
    }
    
    /**
     * 更新像素尺寸（dp/sp 转 px）
     */
    private fun updateDimensions() {
        val density = context.resources.displayMetrics.density
        
        progressRadiusPx = progressRadius * density
        progressStrokeWidthPx = progressStrokeWidth * density
        textSizePx = textSize * density  // 使用 density 代替已废弃的 scaledDensity
        
        trackPaint.strokeWidth = progressStrokeWidthPx
        progressPaint.strokeWidth = progressStrokeWidthPx
        textPaint.textSize = textSizePx
    }
    
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        
        // 计算进度条绘制区域
        val centerX = w / 2f
        val centerY = h / 2f
        val halfStroke = progressStrokeWidthPx / 2f
        
        progressRect.set(
            centerX - progressRadiusPx + halfStroke,
            centerY - progressRadiusPx + halfStroke - textSizePx,  // 稍微往上偏移，给文字留空间
            centerX + progressRadiusPx - halfStroke,
            centerY + progressRadiusPx - halfStroke - textSizePx
        )
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val centerX = width / 2f
        val centerY = height / 2f
        
        // 1. 绘制半透明遮罩
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), overlayPaint)
        
        // 2. 绘制进度条轨道（完整圆环）
        canvas.drawArc(progressRect, 0f, 360f, false, trackPaint)
        
        // 3. 绘制旋转的进度弧
        canvas.drawArc(progressRect, currentRotation, sweepAngle, false, progressPaint)
        
        // 4. 绘制提示文字
        val textY = centerY + progressRadiusPx + textSizePx * 0.5f
        canvas.drawText(calibrationText, centerX, textY, textPaint)
    }
    
    /**
     * 开始旋转动画
     */
    fun startAnimation() {
        if (isAnimating) return
        
        visibility = VISIBLE
        isAnimating = true
        
        rotationAnimator = ValueAnimator.ofFloat(0f, 360f).apply {
            duration = rotationDuration
            interpolator = LinearInterpolator()
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
            
            addUpdateListener { animator ->
                currentRotation = animator.animatedValue as Float
                invalidate()
            }
            
            start()
        }
    }
    
    /**
     * 停止旋转动画并隐藏
     */
    fun stopAnimation() {
        if (!isAnimating) return
        
        isAnimating = false
        rotationAnimator?.cancel()
        rotationAnimator = null
        currentRotation = 0f
        
        visibility = GONE
    }
    
    /**
     * 设置是否显示（会自动管理动画）
     */
    fun setCalibrating(calibrating: Boolean) {
        if (calibrating) {
            startAnimation()
        } else {
            stopAnimation()
        }
    }
    
    /**
     * 是否正在显示动画
     */
    fun isCalibrating(): Boolean = isAnimating
    
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        stopAnimation()
    }
}

// ==================== Compose 桥接 ====================

/**
 * Compose 中使用的校准覆盖层
 * 
 * @param isCalibrating 是否正在校准
 * @param modifier Modifier
 * @param text 提示文字
 */
@Composable
fun CalibrationOverlay(
    isCalibrating: Boolean,
    modifier: Modifier = Modifier,
    text: String = "面捕正在校准中..."
) {
    if (isCalibrating) {
        AndroidView(
            factory = { context ->
                CalibrationOverlayView(context).apply {
                    calibrationText = text
                    startAnimation()
                }
            },
            update = { view ->
                view.calibrationText = text
                view.setCalibrating(isCalibrating)
            },
            modifier = modifier
        )
    }
}
