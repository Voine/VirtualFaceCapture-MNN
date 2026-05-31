package com.example.virtualfacecapture

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import java.io.RandomAccessFile

/**
 * Performance monitor that tracks CPU and memory usage
 * Uses thread CPU time which works on all Android versions
 */
class PerformanceMonitor(context: Context) {
    
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val pid = Process.myPid()
    
    // For CPU calculation using thread time
    private var lastThreadCpuTime = 0L
    private var lastRealTime = 0L
    private val numCores = Runtime.getRuntime().availableProcessors()
    
    /**
     * Get current CPU usage percentage (0-100)
     * Uses thread CPU time which is available on all Android versions
     */
    fun getCpuUsage(): Float {
        return try {
            // Method 1: Use Debug.threadCpuTimeNanos() for current thread
            // This measures how much CPU time the main thread is consuming
            val currentThreadCpuTime = Debug.threadCpuTimeNanos()
            val currentRealTime = SystemClock.elapsedRealtimeNanos()
            
            if (lastThreadCpuTime == 0L) {
                lastThreadCpuTime = currentThreadCpuTime
                lastRealTime = currentRealTime
                return 0f
            }
            
            val cpuTimeDelta = currentThreadCpuTime - lastThreadCpuTime
            val realTimeDelta = currentRealTime - lastRealTime
            
            lastThreadCpuTime = currentThreadCpuTime
            lastRealTime = currentRealTime
            
            if (realTimeDelta > 0) {
                // Calculate percentage (multiply by 100, account for multi-core)
                val usage = (cpuTimeDelta.toFloat() / realTimeDelta.toFloat()) * 100f
                usage.coerceIn(0f, 100f)
            } else {
                0f
            }
        } catch (e: Exception) {
            // Fallback: Try to read from /proc/self/stat (app's own process)
            getProcessCpuUsage()
        }
    }
    
    /**
     * Fallback method: Read CPU usage from /proc/self/stat
     * This reads our own process stats which is allowed
     */
    private var lastProcCpuTime = 0L
    private var lastProcRealTime = 0L
    
    private fun getProcessCpuUsage(): Float {
        return try {
            val statFile = RandomAccessFile("/proc/self/stat", "r")
            val line = statFile.readLine()
            statFile.close()
            
            // Parse the stat file - format: pid (comm) state utime stime ...
            // utime is at index 13, stime is at index 14 (0-based)
            val parts = line.split(" ")
            val utime = parts.getOrNull(13)?.toLongOrNull() ?: 0L
            val stime = parts.getOrNull(14)?.toLongOrNull() ?: 0L
            val procCpuTime = utime + stime
            
            val currentRealTime = SystemClock.elapsedRealtime()
            
            if (lastProcCpuTime == 0L) {
                lastProcCpuTime = procCpuTime
                lastProcRealTime = currentRealTime
                return 0f
            }
            
            val cpuDelta = procCpuTime - lastProcCpuTime
            val timeDelta = currentRealTime - lastProcRealTime
            
            lastProcCpuTime = procCpuTime
            lastProcRealTime = currentRealTime
            
            if (timeDelta > 0) {
                // CPU time is in jiffies (usually 10ms each), convert to percentage
                // Multiply by 10 (jiffy to ms), divide by timeDelta, multiply by 100 for percentage
                val usage = (cpuDelta * 1000f) / timeDelta
                usage.coerceIn(0f, 100f)
            } else {
                0f
            }
        } catch (e: Exception) {
            0f
        }
    }
    
    /**
     * Get current memory usage in MB
     */
    fun getMemoryUsageMB(): Float {
        return try {
            val memInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(memInfo)
            memInfo.totalPss / 1024f // Convert KB to MB
        } catch (e: Exception) {
            // Fallback to Runtime memory info
            val runtime = Runtime.getRuntime()
            (runtime.totalMemory() - runtime.freeMemory()) / (1024f * 1024f)
        }
    }
    
    /**
     * Get total available memory in MB
     */
    fun getTotalMemoryMB(): Float {
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.totalMem / (1024f * 1024f)
    }
    
    /**
     * Get number of CPU cores
     */
    fun getNumCores(): Int = numCores
}

/**
 * Data class to hold performance history
 */
data class PerformanceData(
    val cpuHistory: List<Float> = emptyList(),
    val memoryHistory: List<Float> = emptyList(),
    val currentCpu: Float = 0f,
    val currentMemory: Float = 0f
)

/**
 * Mini line chart for performance visualization
 */
@Composable
fun MiniLineChart(
    data: List<Float>,
    maxValue: Float,
    color: Color,
    modifier: Modifier = Modifier
) {
    if (data.isEmpty()) return
    
    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height
        val stepX = width / (data.size - 1).coerceAtLeast(1)
        
        // Draw background grid line at 50%
        drawLine(
            color = Color.Gray.copy(alpha = 0.3f),
            start = Offset(0f, height / 2),
            end = Offset(width, height / 2),
            strokeWidth = 1f
        )
        
        // Draw the line chart
        if (data.size >= 2) {
            val path = Path()
            data.forEachIndexed { index, value ->
                val x = index * stepX
                val y = height - (value / maxValue * height).coerceIn(0f, height)
                
                if (index == 0) {
                    path.moveTo(x, y)
                } else {
                    path.lineTo(x, y)
                }
            }
            
            drawPath(
                path = path,
                color = color,
                style = Stroke(width = 2f)
            )
        }
        
        // Draw current value dot
        if (data.isNotEmpty()) {
            val lastX = (data.size - 1) * stepX
            val lastY = height - (data.last() / maxValue * height).coerceIn(0f, height)
            drawCircle(
                color = color,
                radius = 3f,
                center = Offset(lastX, lastY)
            )
        }
    }
}

/**
 * Performance chart panel showing CPU and Memory usage
 */
@Composable
fun PerformanceChartPanel(
    modifier: Modifier = Modifier,
    historySize: Int = 30,
    updateIntervalMs: Long = 500
) {
    val context = LocalContext.current
    val performanceMonitor = remember { PerformanceMonitor(context) }
    
    var performanceData by remember { mutableStateOf(PerformanceData()) }
    
    // Update performance data periodically
    LaunchedEffect(Unit) {
        while (true) {
            val cpu = performanceMonitor.getCpuUsage()
            val memory = performanceMonitor.getMemoryUsageMB()
            
            performanceData = performanceData.copy(
                cpuHistory = (performanceData.cpuHistory + cpu).takeLast(historySize),
                memoryHistory = (performanceData.memoryHistory + memory).takeLast(historySize),
                currentCpu = cpu,
                currentMemory = memory
            )
            
            delay(updateIntervalMs)
        }
    }
    
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        // CPU chart
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "CPU",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Cyan,
                modifier = Modifier.width(28.dp)
            )
            
            MiniLineChart(
                data = performanceData.cpuHistory,
                maxValue = 100f,
                color = Color.Cyan,
                modifier = Modifier
                    .width(60.dp)
                    .height(20.dp)
            )
            
            Text(
                text = "${performanceData.currentCpu.toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = when {
                    performanceData.currentCpu > 80 -> Color.Red
                    performanceData.currentCpu > 50 -> Color.Yellow
                    else -> Color.Green
                },
                modifier = Modifier.width(32.dp)
            )
        }
        
        // Memory chart
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                text = "MEM",
                style = MaterialTheme.typography.labelSmall,
                color = Color.Magenta,
                modifier = Modifier.width(28.dp)
            )
            
            MiniLineChart(
                data = performanceData.memoryHistory,
                maxValue = performanceData.memoryHistory.maxOrNull()?.times(1.2f) ?: 500f,
                color = Color.Magenta,
                modifier = Modifier
                    .width(60.dp)
                    .height(20.dp)
            )
            
            Text(
                text = "${performanceData.currentMemory.toInt()}M",
                style = MaterialTheme.typography.labelSmall,
                color = when {
                    performanceData.currentMemory > 400 -> Color.Red
                    performanceData.currentMemory > 200 -> Color.Yellow
                    else -> Color.Green
                },
                modifier = Modifier.width(32.dp)
            )
        }
    }
}
