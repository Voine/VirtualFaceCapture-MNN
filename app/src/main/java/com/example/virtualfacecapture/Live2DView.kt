package com.example.virtualfacecapture

import android.opengl.GLSurfaceView
import android.util.Log
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.live2d.demo.GLRenderer
import com.live2d.demo.JniBridgeJava

/**
 * Live2D View for Jetpack Compose
 * Wraps GLSurfaceView with Live2D rendering
 * 使用 Live2DModelManager 来管理模型加载
 */
@Composable
fun Live2DView(
    modifier: Modifier = Modifier,
    modelManager: Live2DModelManager,
    onModelLoadComplete: () -> Unit = {},
    onModelLoadError: () -> Unit = {}
) {
    var glSurfaceView: GLSurfaceView? by remember { mutableStateOf(null) }
    var isInitialized by remember { mutableStateOf(false) }
    
    DisposableEffect(Unit) {
        onDispose {
            Log.i("Live2DView", "Disposing Live2D view")
            glSurfaceView?.let {
                JniBridgeJava.nativeOnPause()
                JniBridgeJava.nativeOnStop()
            }
        }
    }
    
    AndroidView(
        modifier = modifier,
        factory = { context ->
            Log.i("Live2DView", "Creating GLSurfaceView for Live2D")
            
            // Initialize JniBridge context
            JniBridgeJava.SetContext(context)
            JniBridgeJava.SetActivityInstance(context as android.app.Activity)

            // Set load callback
            JniBridgeJava.setLive2DLoadInterface(object : JniBridgeJava.Live2DLoadInterface {
                override fun onLoadError() {
                    Log.e("Live2DView", "❌ JNI reported: Failed to load Live2D model")
                    onModelLoadError()
                }
                
                override fun onLoadDone() {
                    Log.i("Live2DView", "✓ JNI reported: Live2D model loaded successfully")
                    onModelLoadComplete()
                }
                
                override fun onLoadOneMotion(motionGroup: String?, index: Int, motionName: String?) {
                    Log.d("Live2DView", "Loaded motion: $motionGroup[$index] = $motionName")
                }
                
                override fun onLoadOneExpression(expressionName: String?, index: Int) {
                    Log.d("Live2DView", "Loaded expression: $expressionName[$index]")
                }
            })
            
            // Create GLSurfaceView
            GLSurfaceView(context).apply {
                setEGLContextClientVersion(2)
                setRenderer(GLRenderer())
                renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                
                Log.i("Live2DView", "GLSurfaceView created with OpenGL ES 2.0")
                
                glSurfaceView = this
                
                // Initialize on render thread
                queueEvent {
                    if (!isInitialized) {
                        Log.i("Live2DView", "Initializing Live2D on GL thread...")
                        JniBridgeJava.nativeOnStart()
                        
                        // 使用 ModelManager 加载第一个可用模型
                        Log.i("Live2DView", "Loading first available model via ModelManager...")
                        modelManager.loadFirstAvailableModel { success ->
                            if (success) {
                                Log.i("Live2DView", "✓ Model loaded successfully")
                                
                                // Configure Live2D for face tracking
                                Log.i("Live2DView", "Configuring Live2D for face tracking...")
                                
                                // Disable auto blink - we'll control it via face capture
                                JniBridgeJava.nativeAutoBlinkEyes(false)
                                Log.d("Live2DView", "  - Auto blink: disabled")
                                
                                // Enable manual eye ball control
                                JniBridgeJava.nativeProjectManualEyeBall(true)
                                Log.d("Live2DView", "  - Manual eye ball: enabled")
                                
                                // Enable breath animation
                                JniBridgeJava.nativeBreath(true)
                                Log.d("Live2DView", "  - Breath animation: enabled")
                                
                                Log.i("Live2DView", "✓ Live2D configuration complete")
                            } else {
                                Log.e("Live2DView", "❌ Failed to load model via ModelManager")
                            }
                        }
                        
                        isInitialized = true
                        Log.i("Live2DView", "Live2D view initialization complete")
                    }
                }
            }
        },
        update = { view ->
            // Update when recomposed if needed
        }
    )
}

/**
 * Helper to load a specific Live2D model via ModelManager
 */
fun loadLive2DModel(modelManager: Live2DModelManager, modelName: String, onComplete: (Boolean) -> Unit) {
    val model = modelManager.getAvailableModels().find { it.name == modelName }
    if (model != null) {
        Log.i("Live2DView", "Loading model by name: $modelName")
        modelManager.loadModel(model, onComplete)
    } else {
        Log.e("Live2DView", "Model not found: $modelName")
        onComplete(false)
    }
}

/**
 * Helper to apply expression to Live2D model
 */
fun applyLive2DExpression(expressionName: String) {
    Log.d("Live2DView", "Applying expression: $expressionName")
    JniBridgeJava.nativeApplyExpression(expressionName)
}

/**
 * Helper to apply motion to Live2D model
 */
fun applyLive2DMotion(motionGroup: String, index: Int) {
    Log.d("Live2DView", "Applying motion: $motionGroup[$index]")
    JniBridgeJava.nativeApplyMotion(motionGroup, index)
}