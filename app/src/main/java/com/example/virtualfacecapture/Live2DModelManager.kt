package com.example.virtualfacecapture

import android.content.Context
import android.util.Log
import com.live2d.demo.JniBridgeJava
import java.io.File

/**
 * Live2D 模型管理器
 * 负责从 assets 复制模型到本地，并提供模型加载功能
 */
class Live2DModelManager(private val context: Context) {
    
    companion object {
        private const val TAG = "Live2DModelManager"
        private const val MODELS_ASSETS_PATH = "Live2DModels"
        private const val MODELS_LOCAL_DIR = "live2d_models"
    }
    
    private val modelsLocalPath: String = File(context.filesDir, MODELS_LOCAL_DIR).absolutePath
    private var availableModels: List<ModelInfo> = emptyList()
    private var currentModel: ModelInfo? = null
    
    data class ModelInfo(
        val name: String,
        val assetPath: String,
        val localPath: String,
        val jsonFileName: String
    )
    
    /**
     * 初始化：扫描 assets 中的模型
     */
    fun initialize(onComplete: (List<ModelInfo>) -> Unit) {
        Log.i(TAG, "Initializing Live2D model manager...")
        
        try {
            val modelDirs = context.assets.list(MODELS_ASSETS_PATH) ?: emptyArray()
            Log.i(TAG, "Found ${modelDirs.size} model directories in assets")
            
            val models = mutableListOf<ModelInfo>()
            
            for (modelDir in modelDirs) {
                val assetPath = "$MODELS_ASSETS_PATH/$modelDir"
                val files = context.assets.list(assetPath) ?: continue
                
                // 查找 .model3.json 或 .model.json 文件
                val jsonFile = files.find { 
                    it.endsWith(".model3.json") || it.endsWith(".model.json") 
                }
                
                if (jsonFile != null) {
                    val localPath = File(modelsLocalPath, assetPath).absolutePath
                    val model = ModelInfo(
                        name = modelDir,
                        assetPath = assetPath,
                        localPath = localPath,
                        jsonFileName = jsonFile
                    )
                    models.add(model)
                    Log.i(TAG, "Discovered model: $modelDir (json: $jsonFile)")
                } else {
                    Log.w(TAG, "No .json file found in $modelDir")
                }
            }
            
            availableModels = models
            Log.i(TAG, "Total ${availableModels.size} valid models found")
            onComplete(availableModels)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing model manager", e)
            onComplete(emptyList())
        }
    }
    
    /**
     * 复制模型从 assets 到本地存储
     * @param model 要复制的模型信息
     * @param onProgress 复制进度回调
     * @param onComplete 完成回调 (成功, 本地路径)
     */
    fun copyModelToLocal(
        model: ModelInfo,
        onProgress: ((String) -> Unit)? = null,
        onComplete: (Boolean, String) -> Unit
    ) {
        Log.i(TAG, "Copying model '${model.name}' from assets to local storage...")
        onProgress?.invoke("正在复制模型 ${model.name}...")
        
        val localModelDir = File(model.localPath)

        Log.i(TAG, "Local model path: ${localModelDir.absolutePath}")
        
        // 检查是否已经复制过
        if (localModelDir.exists() && localModelDir.listFiles()?.isNotEmpty() == true) {
            Log.i(TAG, "Model '${model.name}' already exists locally, skipping copy")
            onProgress?.invoke("模型 ${model.name} 已存在")
            onComplete(true, localModelDir.absolutePath)
            return
        }
        
        // 复制模型文件
        context.copyAssets2Local(
            deleteIfExists = true,
            srcPath = model.assetPath,
            desPath = modelsLocalPath
        ) { isSuccess, absPath ->
            if (isSuccess) {
                Log.i(TAG, "✓ Model '${model.name}' copied successfully to: $absPath")
                onProgress?.invoke("模型 ${model.name} 复制完成")
                onComplete(true, localModelDir.absolutePath)
            } else {
                Log.e(TAG, "✗ Failed to copy model '${model.name}'")
                onProgress?.invoke("模型 ${model.name} 复制失败")
                onComplete(false, "")
            }
        }
    }
    
    /**
     * 加载指定模型到 Live2D
     * @param model 模型信息
     * @param onComplete 加载完成回调
     */
    fun loadModel(
        model: ModelInfo,
        onComplete: (Boolean) -> Unit
    ) {
        Log.i(TAG, "Loading model '${model.name}'...")
        
        // 先确保模型已复制到本地
        copyModelToLocal(model) { success, localPath ->
            if (!success) {
                Log.e(TAG, "Cannot load model: copy failed")
                onComplete(false)
                return@copyModelToLocal
            }
            
            try {
                // 传递本地路径给 JniBridge
                Log.i(TAG, "Calling JniBridge.nativeProjectChangeTo()")
                Log.i(TAG, "  Model path: $localPath")
                Log.i(TAG, "  JSON file: ${model.jsonFileName}")
                
                JniBridgeJava.nativeProjectChangeTo("$localPath/", model.jsonFileName)
                val position = getDefaultModelPosition(model.name)
                JniBridgeJava.nativeProjectTransformX(position[0])
                JniBridgeJava.nativeProjectTransformY(position[1])
                JniBridgeJava.nativeProjectScale(position[2])
                currentModel = model
                Log.i(TAG, "✓ Model '${model.name}' loaded successfully")
                onComplete(true)
                
            } catch (e: Exception) {
                Log.e(TAG, "✗ Error loading model '${model.name}'", e)
                onComplete(false)
            }
        }
    }
    
    /**
     * 加载第一个可用的模型
     */
    fun loadFirstAvailableModel(onComplete: (Boolean) -> Unit) {
        if (availableModels.isEmpty()) {
            Log.w(TAG, "No models available to load")
            onComplete(false)
            return
        }
        
        val firstModel = availableModels.find { it.name == "zzzz" } ?: return
        Log.i(TAG, "Loading first available model: ${firstModel.name}")
        loadModel(firstModel, onComplete)
    }
    
    /**
     * 获取所有可用的模型列表
     */
    fun getAvailableModels(): List<ModelInfo> = availableModels
    
    /**
     * 获取当前加载的模型
     */
    fun getCurrentModel(): ModelInfo? = currentModel
    
    /**
     * 清理本地模型缓存
     */
    fun clearLocalCache() {
        Log.i(TAG, "Clearing local model cache...")
        try {
            val localDir = File(modelsLocalPath)
            if (localDir.exists()) {
                FileUtils.deleteDirectory(localDir)
                Log.i(TAG, "✓ Local cache cleared")
            }
        } catch (e: Exception) {
            Log.e(TAG, "✗ Error clearing cache", e)
        }
    }

    fun getDefaultModelPosition(modelName: String): List<Float> {
        val list: List<Float>
        when (modelName) {
            LOCAL_MODEL_YUUKA -> {
                list = listOf(0f, -0.6f, 4f)
            }

            LOCAL_MODEL_ATRI -> {
                list = listOf(0f, -0.6f, 3f)
            }

            LOCAL_MODEL_AMADEUS -> {
                list = listOf(0f, 0f, 2f)
            }

            else -> {
                list = listOf(0f, -0.75f, 3f)
            }
        }
        return list
    }

}

const val LOCAL_MODEL_YUUKA = "Yuuka"
const val LOCAL_MODEL_AMADEUS = "Amadeus"
const val LOCAL_MODEL_ATRI = "ATRI"