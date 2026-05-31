package com.live2d.facecapture

import android.util.Log
import kotlin.math.abs
import kotlin.math.cos

/**
 * 眨眼曲线处理器
 * 
 * 核心思路：
 * 1. 检测到眨眼事件时，用预设的平滑曲线替代原始 EAR 输出
 * 2. 检测到眯眼时，使用曲线平滑过渡到固定值 0.5
 * 3. 消除 Landmark 抖动导致的眼皮颤动
 * 
 * 眨眼曲线参数（基于生理学）：
 * - 闭眼阶段：~80ms（快速）
 * - 保持闭眼：~30ms  
 * - 睁眼阶段：~120ms（稍慢）
 * - 总计：~230ms
 */
typealias BLog = android.util.Log
class BlinkCurveProcessor {
    
    companion object {
        private const val TAG = "BlinkCurveProcessor"
        
        // ============ 默认时间参数（毫秒）============
        private const val DEFAULT_CLOSING_DURATION_MS = 80f
        private const val DEFAULT_CLOSED_DURATION_MS = 30f
        private const val DEFAULT_OPENING_DURATION_MS = 120f
        
        // ============ 眨眼检测参数 ============
        // 连续下降帧数确认眨眼
        private const val DEFAULT_BLINK_CONFIRM_FRAMES = 2
        // 每帧下降率阈值（相对于 baseline）
        private const val DEFAULT_DROP_RATE_THRESHOLD = 0.12f
        // 总下降量阈值（相对于 baseline）
        private const val DEFAULT_MIN_TOTAL_DROP = 0.35f
        
        // ============ 眯眼检测参数 ============
        // 持续低位时间阈值（毫秒），超过此时间判定为眯眼
        private const val DEFAULT_SQUINT_TIME_THRESHOLD_MS = 300f
        // 眯眼时的固定 blink 值
        private const val SQUINT_BLINK_VALUE = 0.5f
        
        // ============ 双眼同步参数 ============
        // 判定为"同时眨眼"的时间窗口（毫秒）
        private const val SYNC_WINDOW_MS = 50L
    }
    
    // ============ 可调参数 ============
    
    /**
     * 眨眼速度系数
     * 
     * 控制整个眨眼动画的时长：
     * - 1.0 = 默认速度（总计约 230ms）
     * - 1.5 = 慢速（总计约 345ms）
     * - 2.0 = 更慢（总计约 460ms）
     * - 0.5 = 快速（总计约 115ms）
     * 
     * 推荐范围: 0.5 ~ 3.0
     */
    var blinkSpeedScale: Float = 1.0f
        set(value) {
            field = value.coerceIn(0.3f, 5.0f)
        }
    
    /**
     * 是否启用双眼同步
     * 当两只眼睛同时检测到眨眼时，同步它们的曲线进度
     */
    var enableBinocularSync: Boolean = true
    
    // ============ 单眼状态 ============
    
    /**
     * 眼睛状态枚举
     * 
     * 状态设计思路：
     * - 眨眼和眯眼共用同一条闭眼曲线
     * - 眨眼：曲线播放到底(1.0)，然后保持，再睁眼
     * - 眯眼：曲线播放到一半(0.5)就停住，固定不动
     */
    enum class EyeState {
        TRACKING,       // 正常跟随 EAR
        BLINK_DETECT,   // 检测到可能的眨眼/眯眼（下降中）
        CLOSING,        // 播放闭眼曲线（眨眼和眯眼共用）
        BLINK_CLOSED,   // 眨眼：保持完全闭眼
        BLINK_OPENING,  // 眨眼：播放睁眼曲线
        SQUINT_HOLD     // 眯眼：固定在0.5不动
    }
    
    /**
     * 单眼处理器
     * 
     * 核心设计：
     * - 只有三种目标状态：睁眼(0)、半睁眼(0.5)、闭眼(1)
     * - 所有状态之间的切换都用平滑曲线
     * - 永远不直接返回原始值，彻底消除抖动
     */
    inner class SingleEyeProcessor(private val eyeName: String) {
        // 当前状态
        var state: EyeState = EyeState.TRACKING
            private set
        
        // 时间戳记录
        var stateStartTimeMs: Long = 0L
            private set
        private var lastUpdateTimeMs: Long = 0L
        
        // EAR 历史（用于检测）
        private val earHistory = ArrayDeque<Float>(5)
        
        // 基线 EAR（睁眼状态的平均值）
        private var baselineEAR: Float = 0.18f
        
        // 眨眼检测计数器
        private var consecutiveDropFrames: Int = 0
        private var dropStartEAR: Float = 0f
        
        // 曲线播放进度（暴露给外部用于同步）
        var curveProgress: Float = 0f
            private set
        
        // ========== 新增：平滑输出系统 ==========
        // 当前实际输出的 blink 值（永远是平滑变化的）
        private var currentOutputValue: Float = 0f
        // 目标 blink 值（0=睁眼, 0.5=半睁眼, 1=闭眼）
        private var targetValue: Float = 0f
        // 过渡起始值
        private var transitionStartValue: Float = 0f
        
        // 眯眼检测
        private var lowEarStartTimeMs: Long = 0L
        private var isInLowEarZone: Boolean = false
        
        // 动态帧率相关
        private var currentFrameIntervalMs: Float = 33.3f // 默认 30fps
        
        /**
         * 更新帧率
         * @param fps 当前帧率
         */
        fun updateFrameRate(fps: Float) {
            currentFrameIntervalMs = if (fps > 0) 1000f / fps else 33.3f
        }
        
        /**
         * 获取动态调整后的检测帧数
         * 基准：30fps 时使用 DEFAULT_BLINK_CONFIRM_FRAMES
         */
        private fun getDynamicConfirmFrames(): Int {
            // 帧率越高，需要更多帧来确认
            val ratio = 33.3f / currentFrameIntervalMs
            return (DEFAULT_BLINK_CONFIRM_FRAMES * ratio).toInt().coerceIn(1, 5)
        }
        
        /**
         * 处理一帧
         * 
         * @param rawBlinkValue 原始的 blink 值 [0, 1]，0=睁眼，1=闭眼
         * @param currentTimeMs 当前时间戳（毫秒）
         * @return 处理后的 blink 值（永远是平滑的）
         */
        fun process(rawBlinkValue: Float, currentTimeMs: Long): Float {
            // 反算 EAR（blink 值越大，EAR 越小）
            val earProxy = 1f - rawBlinkValue
            
            // 更新历史
            earHistory.addLast(earProxy)
            if (earHistory.size > 5) earHistory.removeFirst()
            
            // 计算时间增量
            val deltaMs = if (lastUpdateTimeMs > 0) {
                (currentTimeMs - lastUpdateTimeMs).coerceIn(1, 100)
            } else {
                currentFrameIntervalMs.toLong()
            }
            lastUpdateTimeMs = currentTimeMs
            
            // 根据状态处理（更新目标值和曲线进度）
            when (state) {
                EyeState.TRACKING -> processTracking(earProxy, currentTimeMs)
                EyeState.BLINK_DETECT -> processBlinkDetect(earProxy, currentTimeMs)
                EyeState.CLOSING -> processClosing(currentTimeMs, deltaMs)
                EyeState.BLINK_CLOSED -> processBlinkClosed(currentTimeMs, deltaMs)
                EyeState.BLINK_OPENING -> processBlinkOpening(currentTimeMs, deltaMs)
                EyeState.SQUINT_HOLD -> processSquintHold(earProxy, currentTimeMs)
            }
            
            // 计算当前输出值（基于曲线进度）
            currentOutputValue = calculateSmoothedOutput()
            
            return currentOutputValue
        }
        
        /**
         * 计算平滑输出值
         * 根据当前状态和曲线进度，计算实际输出
         */
        private fun calculateSmoothedOutput(): Float {
            return when (state) {
                EyeState.TRACKING -> {
                    // 睁眼状态，固定返回 0
                    0f
                }
                EyeState.BLINK_DETECT -> {
                    // 检测中，保持在睁眼状态（0），等待确认
                    0f
                }
                EyeState.CLOSING -> {
                    // 闭眼过程中，使用曲线
                    val curveValue = closingCurve(curveProgress)
                    if (isBlinkNotSquint) {
                        // 眨眼：0 -> 1
                        curveValue
                    } else {
                        // 眯眼：0 -> 0.5
                        curveValue * SQUINT_BLINK_VALUE
                    }
                }
                EyeState.BLINK_CLOSED -> {
                    // 完全闭眼，固定返回 1
                    1f
                }
                EyeState.BLINK_OPENING -> {
                    // 睁眼过程中，使用曲线 1 -> 0
                    openingCurve(curveProgress)
                }
                EyeState.SQUINT_HOLD -> {
                    // 眯眼保持，固定返回 0.5
                    SQUINT_BLINK_VALUE
                }
            }
        }
        
        /**
         * TRACKING 状态：睁眼状态，检测是否开始眨眼/眯眼
         */
        private fun processTracking(earProxy: Float, currentTimeMs: Long) {
            // 更新基线（使用高位 EAR 的移动平均）
            if (earProxy > baselineEAR * 0.85f) {
                baselineEAR = baselineEAR * 0.95f + earProxy * 0.05f
            }
            
            // 检测快速下降（可能是眨眼）
            if (earHistory.size >= 2) {
                val prev = earHistory[earHistory.size - 2]
                val curr = earHistory.last()
                val dropRate = (prev - curr) / baselineEAR
                
                if (dropRate > DEFAULT_DROP_RATE_THRESHOLD) {
                    consecutiveDropFrames = 1
                    dropStartEAR = prev
                    transitionTo(EyeState.BLINK_DETECT, currentTimeMs)
                    return
                }
            }
            
            // 检测眯眼（持续低位）
            val lowThreshold = baselineEAR * 0.65f
            if (earProxy < lowThreshold) {
                if (!isInLowEarZone) {
                    isInLowEarZone = true
                    lowEarStartTimeMs = currentTimeMs
                }
                // 如果持续低位超过阈值，进入眯眼
                if (currentTimeMs - lowEarStartTimeMs > DEFAULT_SQUINT_TIME_THRESHOLD_MS) {
                    Log.d(TAG, "$eyeName: Squint detected from TRACKING")
                    isBlinkNotSquint = false
                    transitionTo(EyeState.CLOSING, currentTimeMs)
                }
            } else {
                isInLowEarZone = false
            }
        }
        
        /**
         * BLINK_DETECT 状态：检测中，等待确认眨眼或眯眼
         * 在此状态下输出保持为 0（睁眼），不返回原始值
         */
        @Suppress("UNUSED_PARAMETER")
        private fun processBlinkDetect(earProxy: Float, currentTimeMs: Long) {
            val timeInState = currentTimeMs - stateStartTimeMs
            
            if (earHistory.size >= 2) {
                val prev = earHistory[earHistory.size - 2]
                val curr = earHistory.last()
                val dropRate = (prev - curr) / baselineEAR
                val totalDrop = (dropStartEAR - curr) / baselineEAR
                
                // 快速大幅下降 → 确认眨眼
                if (totalDrop > DEFAULT_MIN_TOTAL_DROP * 1.2f) {
                    Log.d(TAG, "$eyeName: Blink confirmed by large drop!")
                    isBlinkNotSquint = true
                    transitionTo(EyeState.CLOSING, currentTimeMs)
                    return
                }
                
                if (dropRate > DEFAULT_DROP_RATE_THRESHOLD * 0.5f) {
                    consecutiveDropFrames++
                    
                    if (consecutiveDropFrames >= getDynamicConfirmFrames() && 
                        totalDrop > DEFAULT_MIN_TOTAL_DROP) {
                        Log.d(TAG, "$eyeName: Blink confirmed!")
                        isBlinkNotSquint = true
                        transitionTo(EyeState.CLOSING, currentTimeMs)
                        return
                    }
                } else if (dropRate < -DEFAULT_DROP_RATE_THRESHOLD * 0.3f && totalDrop < DEFAULT_MIN_TOTAL_DROP * 0.5f) {
                    // EAR 回升且下降量不够，取消检测，回到睁眼
                    Log.d(TAG, "$eyeName: BLINK_DETECT cancelled")
                    transitionTo(EyeState.TRACKING, currentTimeMs)
                    consecutiveDropFrames = 0
                    return
                }
            }
            
            // 超时判定
            if (timeInState > DEFAULT_SQUINT_TIME_THRESHOLD_MS) {
                val currentEar = earHistory.lastOrNull() ?: baselineEAR
                if (currentEar < baselineEAR * 0.4f) {
                    // 眼睛很闭 → 慢速眨眼
                    Log.d(TAG, "$eyeName: Slow blink confirmed")
                    isBlinkNotSquint = true
                    transitionTo(EyeState.CLOSING, currentTimeMs)
                } else {
                    // 眼睛半闭 → 眯眼
                    Log.d(TAG, "$eyeName: Squint detected")
                    isBlinkNotSquint = false
                    transitionTo(EyeState.CLOSING, currentTimeMs)
                }
            }
        }
        
        // 标记当前 CLOSING 是眨眼(true)还是眯眼(false)
        private var isBlinkNotSquint: Boolean = true
        
        /**
         * CLOSING 状态：播放闭眼曲线
         */
        private fun processClosing(currentTimeMs: Long, deltaMs: Long) {
            val duration = DEFAULT_CLOSING_DURATION_MS * blinkSpeedScale
            curveProgress += deltaMs / duration
            
            if (isBlinkNotSquint) {
                // 眨眼：播放到底
                if (curveProgress >= 1f) {
                    curveProgress = 0f
                    transitionTo(EyeState.BLINK_CLOSED, currentTimeMs)
                }
            } else {
                // 眯眼：曲线值到达 0.5 对应的进度就停
                // closingCurve(t) = 0.5 时的 t 约为 0.5
                if (curveProgress >= 1f) {
                    Log.d(TAG, "$eyeName: Squint reached target")
                    transitionTo(EyeState.SQUINT_HOLD, currentTimeMs)
                }
            }
        }
        
        /**
         * BLINK_CLOSED 状态：保持闭眼
         * 
         * 改进：检测原始 EAR 值来区分"快速眨眼"和"持续闭眼"
         * - 如果 EAR 持续很低 → 用户想保持闭眼 → 继续保持
         * - 如果 EAR 开始回升 → 用户想睁眼 → 开始睁眼曲线
         */
        private fun processBlinkClosed(currentTimeMs: Long, deltaMs: Long) {
            val duration = DEFAULT_CLOSED_DURATION_MS * blinkSpeedScale
            curveProgress += deltaMs / duration
            
            // 检测用户是否想睁眼（EAR 回升）
            val currentEar = earHistory.lastOrNull() ?: 0f
            val isEyeStillClosed = currentEar < baselineEAR * 0.5f
            
            if (curveProgress >= 1f) {
                if (isEyeStillClosed) {
                    // 用户仍在闭眼，保持 BLINK_CLOSED 状态，不自动睁眼
                    // 重置进度，继续等待
                    curveProgress = 0f
                    Log.v(TAG, "$eyeName: Eye still closed, holding...")
                } else {
                    // 用户想睁眼了，开始睁眼曲线
                    curveProgress = 0f
                    transitionTo(EyeState.BLINK_OPENING, currentTimeMs)
                }
            }
        }
        
        /**
         * BLINK_OPENING 状态：播放睁眼曲线
         */
        private fun processBlinkOpening(currentTimeMs: Long, deltaMs: Long) {
            val duration = DEFAULT_OPENING_DURATION_MS * blinkSpeedScale
            curveProgress += deltaMs / duration
            
            if (curveProgress >= 1f) {
                curveProgress = 0f
                consecutiveDropFrames = 0
                transitionTo(EyeState.TRACKING, currentTimeMs)
            }
        }
        
        /**
         * SQUINT_HOLD 状态：眯眼保持
         */
        private fun processSquintHold(earProxy: Float, currentTimeMs: Long) {
            // 检测是否恢复睁眼
            if (earProxy > baselineEAR * 0.8f) {
                Log.d(TAG, "$eyeName: Squint ended")
                // 从 0.5 平滑回到 0，使用睁眼曲线
                transitionTo(EyeState.BLINK_OPENING, currentTimeMs)
                // 从半闭开始睁眼，进度设为 0.5
                curveProgress = 0.5f
                isInLowEarZone = false
                return
            }
            
            // 检测是否变成眨眼（继续闭眼）
            if (earHistory.size >= 2) {
                val prev = earHistory[earHistory.size - 2]
                val curr = earHistory.last()
                val dropRate = (prev - curr) / baselineEAR
                
                if (dropRate > DEFAULT_DROP_RATE_THRESHOLD && curr < baselineEAR * 0.3f) {
                    Log.d(TAG, "$eyeName: Squint -> Blink")
                    isBlinkNotSquint = true
                    transitionTo(EyeState.CLOSING, currentTimeMs)
                    // 从 0.5 开始继续闭眼
                    curveProgress = 0.5f
                }
            }
        }
        
        /**
         * 状态转换
         */
        private fun transitionTo(newState: EyeState, currentTimeMs: Long) {
            if (state != newState) {
                Log.v(TAG, "$eyeName: $state -> $newState")
                state = newState
                stateStartTimeMs = currentTimeMs
                if (newState != EyeState.BLINK_OPENING || state != EyeState.SQUINT_HOLD) {
                    // 除了从眯眼到睁眼（需要保留进度），其他情况重置进度
                    if (state != EyeState.SQUINT_HOLD) {
                        curveProgress = 0f
                    }
                }
            }
        }
        
        /**
         * 重置状态
         */
        fun reset() {
            state = EyeState.TRACKING
            stateStartTimeMs = 0L
            lastUpdateTimeMs = 0L
            earHistory.clear()
            baselineEAR = 0.18f
            consecutiveDropFrames = 0
            dropStartEAR = 0f
            curveProgress = 0f
            currentOutputValue = 0f
            targetValue = 0f
            transitionStartValue = 0f
            lowEarStartTimeMs = 0L
            isInLowEarZone = false
            isBlinkNotSquint = true
        }
    }
    
    // ============ 曲线函数 ============
    
    /**
     * 闭眼曲线 (0→1)
     * 
     * 使用正弦函数实现：
     * - 开始（睁眼→半闭）：慢慢加速
     * - 中间（半闭区域）：最快
     * - 结束（半闭→闭眼）：慢慢减速
     * 
     * 公式：(1 - cos(t * π)) / 2
     * 这是 sin 函数的积分形式，产生 S 形曲线
     * 
     * t=0.0 → 0.0 (睁眼)
     * t=0.25 → 0.146 (开始慢)
     * t=0.5 → 0.5 (中间，速度最快)
     * t=0.75 → 0.854 (结束慢)
     * t=1.0 → 1.0 (闭眼)
     */
    private fun closingCurve(t: Float): Float {
        val clampedT = t.coerceIn(0f, 1f)
        // (1 - cos(t * π)) / 2 = 正弦 ease-in-out
        return (1f - cos(clampedT * Math.PI).toFloat()) / 2f
    }
    
    /**
     * 睁眼曲线 (1→0)
     * 
     * 同样使用正弦函数，实现对称的 S 形曲线
     * 
     * t=0.0 → 1.0 (闭眼)
     * t=0.25 → 0.854 (开始慢)
     * t=0.5 → 0.5 (中间，速度最快)
     * t=0.75 → 0.146 (结束慢)
     * t=1.0 → 0.0 (睁眼)
     */
    private fun openingCurve(t: Float): Float {
        val clampedT = t.coerceIn(0f, 1f)
        // 1 - closingCurve(t) 实现对称效果
        return (1f + cos(clampedT * Math.PI).toFloat()) / 2f
    }
    
    // ============ 双眼处理器 ============
    
    val leftEye = SingleEyeProcessor("LeftEye")
    val rightEye = SingleEyeProcessor("RightEye")
    
    // 是否启用
    var isEnabled: Boolean = true
        set(value) {
            BLog.i(TAG, "BlinkCurveProcessor isEnabled set to $value")
            field = value
            if (value) {
                reset()
            }
        }
    
    /**
     * 更新帧率
     * @param fps 当前帧率 (30-60)
     */
    fun updateFrameRate(fps: Float) {
        leftEye.updateFrameRate(fps)
        rightEye.updateFrameRate(fps)
    }
    
    /**
     * 处理双眼 blink 值
     * 
     * @param leftBlinkRaw 左眼原始 blink 值
     * @param rightBlinkRaw 右眼原始 blink 值
     * @param currentTimeMs 当前时间戳
     * @return Pair(leftBlink, rightBlink) 处理后的值
     */
    fun process(leftBlinkRaw: Float, rightBlinkRaw: Float, currentTimeMs: Long = System.currentTimeMillis()): Pair<Float, Float> {
        if (!isEnabled) {
            return Pair(leftBlinkRaw, rightBlinkRaw)
        }
        
        // 先独立处理两只眼睛
        val leftResult = leftEye.process(leftBlinkRaw, currentTimeMs)
        val rightResult = rightEye.process(rightBlinkRaw, currentTimeMs)
        
        // 双眼同步逻辑
        if (enableBinocularSync) {
            val syncedResults = synchronizeBlink(leftResult, rightResult, currentTimeMs)
            return syncedResults
        }
        
        return Pair(leftResult, rightResult)
    }
    
    /**
     * 双眼同步逻辑
     * 
     * 当两只眼睛都在眨眼状态（CLOSING/CLOSED/OPENING）时，
     * 将两只眼睛的 blink 值拉齐到较大的那个值（更闭的那个）
     */
    private fun synchronizeBlink(leftResult: Float, rightResult: Float, currentTimeMs: Long): Pair<Float, Float> {
        val leftState = leftEye.state
        val rightState = rightEye.state
        
        // 定义眨眼相关状态（不包含 SQUINT_HOLD，眯眼不同步）
        val blinkStates = setOf(
            EyeState.CLOSING,
            EyeState.BLINK_CLOSED,
            EyeState.BLINK_OPENING
        )
        
        val leftInBlink = leftState in blinkStates
        val rightInBlink = rightState in blinkStates
        
        // 如果两只眼睛都在眨眼状态，进行同步
        if (leftInBlink && rightInBlink) {
            // 检查两只眼睛是否在时间窗口内进入眨眼状态（判定为同时眨眼）
            val timeDiff = abs(leftEye.stateStartTimeMs - rightEye.stateStartTimeMs)
            
            if (timeDiff <= SYNC_WINDOW_MS * 3) {  // 放宽时间窗口，因为状态可能不同步
                // 取较大的值（更闭的那个眼睛）
                val syncedValue = maxOf(leftResult, rightResult)
                Log.v(TAG, "Binocular sync: L=$leftResult, R=$rightResult -> $syncedValue")
                return Pair(syncedValue, syncedValue)
            }
        }
        
        // 如果一只眼睛刚进入眨眼，另一只也在检测中，同步触发
        if (leftState == EyeState.CLOSING && rightState == EyeState.BLINK_DETECT) {
            val timeDiff = currentTimeMs - rightEye.stateStartTimeMs
            if (timeDiff <= SYNC_WINDOW_MS * 2) {
                // 右眼也触发眨眼
                Log.d(TAG, "Sync trigger: Right eye follows left")
                return Pair(leftResult, leftResult)
            }
        }
        
        if (rightState == EyeState.CLOSING && leftState == EyeState.BLINK_DETECT) {
            val timeDiff = currentTimeMs - leftEye.stateStartTimeMs
            if (timeDiff <= SYNC_WINDOW_MS * 2) {
                // 左眼也触发眨眼
                Log.d(TAG, "Sync trigger: Left eye follows right")
                return Pair(rightResult, rightResult)
            }
        }
        
        return Pair(leftResult, rightResult)
    }
    
    /**
     * 重置所有状态
     */
    fun reset() {
        leftEye.reset()
        rightEye.reset()
    }
    
    /**
     * 获取当前状态（用于调试）
     */
    fun getDebugInfo(): String {
        return "L:${leftEye.state.name.take(4)} R:${rightEye.state.name.take(4)} spd=${blinkSpeedScale}"
    }
}
