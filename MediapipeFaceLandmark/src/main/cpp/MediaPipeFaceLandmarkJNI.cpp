#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// Use new modular implementation
#include "mediapipe_face.h"

#define LOG_TAG "MediaPipeFaceJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace mediapipe_face;

// Global instances for face detection, landmark detection, and blendshape prediction
static std::unique_ptr<FaceDetector> g_faceDetector = nullptr;
static std::unique_ptr<FaceLandmarker> g_faceLandmarker = nullptr;
static std::unique_ptr<BlendShapePredictor> g_blendshapePredictor = nullptr;

// Cached head pose for compensation
static HeadPose g_lastHeadPose;
static bool g_hasHeadPose = false;

/**
 * Initialize face detector, landmarker, and blendshape predictor models
 * @param assetManager Android AssetManager
 * @param detectorModelPath Path to face detector model in assets
 * @param landmarkerModelPath Path to face landmarker model in assets
 * @param blendshapeModelPath Path to blendshape model in assets (optional, can be null)
 * @param numThreads Number of threads for inference
 * @return true if successful
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeInit(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager,
        jstring detectorModelPath,
        jstring landmarkerModelPath,
        jstring blendshapeModelPath,
        jint numThreads) {
    
    LOGI("Initializing MediaPipe Face Detection with BlendShape support");
    
    const char* detectorPath = env->GetStringUTFChars(detectorModelPath, nullptr);
    const char* landmarkerPath = env->GetStringUTFChars(landmarkerModelPath, nullptr);
    const char* blendshapePath = blendshapeModelPath ? 
                                  env->GetStringUTFChars(blendshapeModelPath, nullptr) : nullptr;
    
    // Clean up existing instances
    g_faceDetector.reset();
    g_faceLandmarker.reset();
    g_blendshapePredictor.reset();
    g_hasHeadPose = false;
    
    // Create detector
    g_faceDetector = std::make_unique<FaceDetector>();
    if (!g_faceDetector->loadModel(detectorPath, numThreads)) {
        LOGE("Failed to load face detector: %s", g_faceDetector->lastError().c_str());
        env->ReleaseStringUTFChars(detectorModelPath, detectorPath);
        env->ReleaseStringUTFChars(landmarkerModelPath, landmarkerPath);
        if (blendshapePath) env->ReleaseStringUTFChars(blendshapeModelPath, blendshapePath);
        return JNI_FALSE;
    }
    LOGI("Face detector loaded successfully");
    
    // Create landmarker
    g_faceLandmarker = std::make_unique<FaceLandmarker>();
    if (!g_faceLandmarker->loadModel(landmarkerPath, numThreads)) {
        LOGE("Failed to load face landmarker: %s", g_faceLandmarker->lastError().c_str());
        env->ReleaseStringUTFChars(detectorModelPath, detectorPath);
        env->ReleaseStringUTFChars(landmarkerModelPath, landmarkerPath);
        if (blendshapePath) env->ReleaseStringUTFChars(blendshapeModelPath, blendshapePath);
        return JNI_FALSE;
    }
    LOGI("Face landmarker loaded successfully");
    
    // Create blendshape predictor (optional)
    if (blendshapePath) {
        g_blendshapePredictor = std::make_unique<BlendShapePredictor>();
        if (!g_blendshapePredictor->loadModel(blendshapePath, numThreads)) {
            LOGE("Failed to load blendshape predictor: %s", g_blendshapePredictor->lastError().c_str());
            // Continue without blendshape support
            g_blendshapePredictor.reset();
            LOGI("BlendShape prediction will be disabled");
        } else {
            LOGI("BlendShape predictor loaded successfully");
        }
    }
    
    env->ReleaseStringUTFChars(detectorModelPath, detectorPath);
    env->ReleaseStringUTFChars(landmarkerModelPath, landmarkerPath);
    if (blendshapePath) env->ReleaseStringUTFChars(blendshapeModelPath, blendshapePath);
    
    return JNI_TRUE;
}

/**
 * Set dump directory for debug frame capture
 * The first frame passed to detector will be saved as PPM image
 * @param dumpDir Directory path where to save the debug frame
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetDumpDirectory(
        JNIEnv* env,
        jobject /* this */,
        jstring dumpDir) {
    
    if (dumpDir == nullptr) {
        LOGI("Dump directory set to null, disabling frame dump");
        FaceDetector::setDumpDirectory("");
        return;
    }
    
    const char* dir = env->GetStringUTFChars(dumpDir, nullptr);
    LOGI("Setting dump directory to: %s", dir);
    FaceDetector::setDumpDirectory(dir);
    env->ReleaseStringUTFChars(dumpDir, dir);
}

/**
 * Detect face landmarks from RGB image data
 * @param imageData RGB image data (width * height * 3 bytes)
 * @param width Image width
 * @param height Image height
 * @param scoreThreshold Face detection score threshold
 * @param iouThreshold Face detection NMS IOU threshold
 * @param presenceThreshold Landmark presence threshold
 * @return Float array of landmarks [x1,y1,z1, x2,y2,z2, ...] or null if no face detected
 *         Coordinates are normalized to [0, 1] relative to image size
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectLandmarks(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height,
        jfloat scoreThreshold,
        jfloat iouThreshold,
        jfloat presenceThreshold) {
    
    if (!g_faceDetector || !g_faceLandmarker) {
        LOGE("Models not initialized. Call nativeInit first.");
        return nullptr;
    }
    
    // Get image data
    jbyte* imgBytes = env->GetByteArrayElements(imageData, nullptr);
    uint8_t* imgData = reinterpret_cast<uint8_t*>(imgBytes);
    
    // Step 1: Detect faces
    auto faces = g_faceDetector->detect(imgData, width, height, scoreThreshold, iouThreshold);
    
    if (faces.empty()) {
        LOGI("No face detected");
        env->ReleaseByteArrayElements(imageData, imgBytes, JNI_ABORT);
        return nullptr;
    }
    
    LOGI("Detected %zu face(s), processing first face", faces.size());
    
    // Step 2: Get landmarks for first face
    FaceDetection& face = faces[0];
    auto result = g_faceLandmarker->detect(imgData, width, height, face, presenceThreshold);
    
    env->ReleaseByteArrayElements(imageData, imgBytes, JNI_ABORT);
    
    if (!result.isValid()) {
        LOGI("Failed to detect landmarks (presence score: %.3f)", result.presence_score);
        return nullptr;
    }
    
    LOGI("Detected %zu landmarks with presence score %.3f", 
         result.landmarks.size(), result.presence_score);
    
    // Step 3: Convert landmarks to float array [x,y,z, x,y,z, ...]
    // Landmarks are already normalized to [0, 1] by the detector
    size_t numLandmarks = result.landmarks.size();
    jfloatArray landmarksArray = env->NewFloatArray(numLandmarks * 3);
    
    std::vector<float> landmarksData;
    landmarksData.reserve(numLandmarks * 3);
    
    for (const auto& lm : result.landmarks) {
        landmarksData.push_back(lm.x);
        landmarksData.push_back(lm.y);
        landmarksData.push_back(lm.z);
    }
    
    env->SetFloatArrayRegion(landmarksArray, 0, numLandmarks * 3, landmarksData.data());
    
    return landmarksArray;
}

/**
 * Stage 1: Detect face bounding box only (fast, ~4ms)
 * Separate from landmark detection for pipeline parallelism
 * @return Float array [x1, y1, x2, y2, score] or null if no face
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectFace(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height,
        jfloat scoreThreshold,
        jfloat iouThreshold) {
    
    if (!g_faceDetector) {
        LOGE("Face detector not initialized");
        return nullptr;
    }
    
    jbyte* imgBytes = env->GetByteArrayElements(imageData, nullptr);
    uint8_t* imgData = reinterpret_cast<uint8_t*>(imgBytes);
    
    // Detect faces
    auto faces = g_faceDetector->detect(imgData, width, height, scoreThreshold, iouThreshold);
    
    env->ReleaseByteArrayElements(imageData, imgBytes, JNI_ABORT);
    
    if (faces.empty()) {
        return nullptr;
    }
    
    // Return first face as [x1, y1, x2, y2, score]
    FaceDetection& face = faces[0];
    jfloatArray result = env->NewFloatArray(5);
    float boxData[5] = { face.x1, face.y1, face.x2, face.y2, face.score };
    env->SetFloatArrayRegion(result, 0, 5, boxData);
    
    return result;
}

/**
 * Stage 2: Detect landmarks using pre-computed bounding box (slower, ~24ms)
 * Uses provided bounding box instead of running face detection
 * @param boxX1, boxY1, boxX2, boxY2 Normalized bounding box coordinates [0,1]
 * @return Float array of landmarks [x,y,z, ...] or null on failure
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectLandmarksWithBox(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray imageData,
        jint width,
        jint height,
        jfloat boxX1,
        jfloat boxY1,
        jfloat boxX2,
        jfloat boxY2,
        jfloat presenceThreshold) {
    
    if (!g_faceLandmarker) {
        LOGE("Face landmarker not initialized");
        return nullptr;
    }
    
    jbyte* imgBytes = env->GetByteArrayElements(imageData, nullptr);
    uint8_t* imgData = reinterpret_cast<uint8_t*>(imgBytes);
    
    // Create FaceDetection from provided box
    FaceDetection face;
    face.x1 = boxX1;
    face.y1 = boxY1;
    face.x2 = boxX2;
    face.y2 = boxY2;
    face.score = 1.0f;  // Assume valid box
    
    // Run landmark detection
    auto result = g_faceLandmarker->detect(imgData, width, height, face, presenceThreshold);
    
    env->ReleaseByteArrayElements(imageData, imgBytes, JNI_ABORT);
    
    if (!result.isValid()) {
        LOGI("Landmark detection failed with provided box");
        return nullptr;
    }
    
    // Convert landmarks to float array
    size_t numLandmarks = result.landmarks.size();
    jfloatArray landmarksArray = env->NewFloatArray(numLandmarks * 3);
    
    std::vector<float> landmarksData;
    landmarksData.reserve(numLandmarks * 3);
    
    for (const auto& lm : result.landmarks) {
        landmarksData.push_back(lm.x);
        landmarksData.push_back(lm.y);
        landmarksData.push_back(lm.z);
    }
    
    env->SetFloatArrayRegion(landmarksArray, 0, numLandmarks * 3, landmarksData.data());
    
    return landmarksArray;
}

/**
 * Estimate head pose from landmarks
 * @param landmarks Float array of landmarks [x,y,z, ...]
 * @param imageWidth Image width
 * @param imageHeight Image height
 * @return Float array [pitch, yaw, roll] in degrees
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeEstimateHeadPose(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray landmarks,
        jint imageWidth,
        jint imageHeight) {
    
    if (!g_faceLandmarker) {
        LOGE("Landmarker not initialized");
        return nullptr;
    }
    
    // Convert landmarks array to vector
    jsize numFloats = env->GetArrayLength(landmarks);
    jfloat* landmarksData = env->GetFloatArrayElements(landmarks, nullptr);
    
    std::vector<FaceLandmark> landmarksVec;
    for (jsize i = 0; i < numFloats; i += 3) {
        FaceLandmark lm;
        lm.x = landmarksData[i];
        lm.y = landmarksData[i + 1];
        lm.z = landmarksData[i + 2];
        landmarksVec.push_back(lm);
    }
    
    env->ReleaseFloatArrayElements(landmarks, landmarksData, JNI_ABORT);
    
    // Estimate head pose
    HeadPose pose = FaceLandmarker::estimateHeadPose(
        landmarksVec, imageWidth, imageHeight);
    
    // Return [pitch, yaw, roll]
    jfloatArray poseArray = env->NewFloatArray(3);
    float poseData[3] = { pose.pitch, pose.yaw, pose.roll };
    env->SetFloatArrayRegion(poseArray, 0, 3, poseData);
    
    return poseArray;
}

/**
 * Release native resources
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeRelease(
        JNIEnv* env,
        jobject /* this */) {
    
    LOGI("Releasing MediaPipe Face Detection resources");
    
    g_faceDetector.reset();
    g_faceLandmarker.reset();
    g_blendshapePredictor.reset();
    
    g_hasHeadPose = false;
}

/**
 * Predict blendshapes from face landmarks
 * @param landmarks Float array of landmarks [x,y,z, ...] (478 landmarks * 3 = 1434 floats)
 * @param imageWidth Image width (for denormalization)
 * @param imageHeight Image height (for denormalization)
 * @param useHeadPoseCompensation Whether to apply head rotation compensation
 * @return Float array of 52 blendshape values [0-1], or null if prediction failed
 *         Values are ordered according to MediaPipe's BlendShape indices (ARKit compatible)
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativePredictBlendshapes(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray landmarks,
        jint imageWidth,
        jint imageHeight,
        jboolean useHeadPoseCompensation) {
    
    if (!g_blendshapePredictor) {
        LOGE("BlendShape predictor not initialized");
        return nullptr;
    }
    
    // Convert landmarks array to vector
    jsize numFloats = env->GetArrayLength(landmarks);
    if (numFloats < 1404) {  // 468 * 3 minimum
        LOGE("Insufficient landmarks for BlendShape prediction: %d floats", numFloats);
        return nullptr;
    }
    
    jfloat* landmarksData = env->GetFloatArrayElements(landmarks, nullptr);
    
    std::vector<FaceLandmark> landmarksVec;
    landmarksVec.reserve(numFloats / 3);
    for (jsize i = 0; i < numFloats; i += 3) {
        FaceLandmark lm;
        lm.x = landmarksData[i];
        lm.y = landmarksData[i + 1];
        lm.z = landmarksData[i + 2];
        landmarksVec.push_back(lm);
    }
    
    // Update head pose if compensation is enabled
    HeadPose* posePtr = nullptr;
    if (useHeadPoseCompensation && g_faceLandmarker) {
        g_lastHeadPose = FaceLandmarker::estimateHeadPose(landmarksVec, imageWidth, imageHeight);
        g_hasHeadPose = g_lastHeadPose.isValid();
        if (g_hasHeadPose) {
            posePtr = &g_lastHeadPose;
            LOGI("Head pose: pitch=%.2f, yaw=%.2f, roll=%.2f", 
                 g_lastHeadPose.pitch, g_lastHeadPose.yaw, g_lastHeadPose.roll);
        }
    }
    
    env->ReleaseFloatArrayElements(landmarks, landmarksData, JNI_ABORT);
    
    // Predict blendshapes
    BlendShapeResult result = g_blendshapePredictor->predict(
        landmarksVec, imageWidth, imageHeight, posePtr);
    
    if (!result.valid) {
        LOGE("BlendShape prediction failed: %s", g_blendshapePredictor->lastError().c_str());
        return nullptr;
    }
    
    // Return compensated blendshape values
    jfloatArray blendshapeArray = env->NewFloatArray(BS_COUNT);
    env->SetFloatArrayRegion(blendshapeArray, 0, BS_COUNT, result.compensated);
    
    LOGI("BlendShape prediction successful: eyeBlink L=%.2f R=%.2f, jawOpen=%.2f", 
         result.compensated[BS_EYE_BLINK_LEFT], 
         result.compensated[BS_EYE_BLINK_RIGHT],
         result.compensated[BS_JAW_OPEN]);
    
    return blendshapeArray;
}

/**
 * Get raw blendshapes without head pose compensation
 * @param landmarks Float array of landmarks [x,y,z, ...]
 * @param imageWidth Image width
 * @param imageHeight Image height
 * @return Float array of 52 raw blendshape values [0-1]
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativePredictBlendshapesRaw(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray landmarks,
        jint imageWidth,
        jint imageHeight) {
    
    if (!g_blendshapePredictor) {
        LOGE("BlendShape predictor not initialized");
        return nullptr;
    }
    
    jsize numFloats = env->GetArrayLength(landmarks);
    if (numFloats < 1404) {
        LOGE("Insufficient landmarks for BlendShape prediction");
        return nullptr;
    }
    
    jfloat* landmarksData = env->GetFloatArrayElements(landmarks, nullptr);
    
    std::vector<FaceLandmark> landmarksVec;
    landmarksVec.reserve(numFloats / 3);
    for (jsize i = 0; i < numFloats; i += 3) {
        FaceLandmark lm;
        lm.x = landmarksData[i];
        lm.y = landmarksData[i + 1];
        lm.z = landmarksData[i + 2];
        landmarksVec.push_back(lm);
    }
    
    env->ReleaseFloatArrayElements(landmarks, landmarksData, JNI_ABORT);
    
    // Predict without head pose compensation
    BlendShapeResult result = g_blendshapePredictor->predict(
        landmarksVec, imageWidth, imageHeight, nullptr);
    
    if (!result.valid) {
        LOGE("BlendShape prediction failed");
        return nullptr;
    }
    
    // Return raw (uncompensated) values
    jfloatArray blendshapeArray = env->NewFloatArray(BS_COUNT);
    env->SetFloatArrayRegion(blendshapeArray, 0, BS_COUNT, result.values);
    
    return blendshapeArray;
}

/**
 * Get the last estimated head pose
 * @return Float array [pitch, yaw, roll] in degrees, or null if no pose available
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeGetLastHeadPose(
        JNIEnv* env,
        jobject /* this */) {
    
    if (!g_hasHeadPose) {
        return nullptr;
    }
    
    jfloatArray poseArray = env->NewFloatArray(6);
    float poseData[6] = {
        g_lastHeadPose.pitch, 
        g_lastHeadPose.yaw, 
        g_lastHeadPose.roll,
        g_lastHeadPose.translation_x,
        g_lastHeadPose.translation_y,
        g_lastHeadPose.translation_z
    };
    env->SetFloatArrayRegion(poseArray, 0, 6, poseData);
    
    return poseArray;
}

/**
 * Get BlendShape name by index
 * @param index BlendShape index (0-51)
 * @return BlendShape name string
 */
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeGetBlendshapeName(
        JNIEnv* env,
        jobject /* this */,
        jint index) {
    
    if (index < 0 || index >= BS_COUNT) {
        return env->NewStringUTF("unknown");
    }
    
    return env->NewStringUTF(getBlendShapeName(static_cast<BlendShapeIndex>(index)));
}

/**
 * Check if BlendShape prediction is available
 * @return true if BlendShape model is loaded
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeIsBlendshapeAvailable(
        JNIEnv* env,
        jobject /* this */) {
    
    return g_blendshapePredictor != nullptr ? JNI_TRUE : JNI_FALSE;
}

/**
 * Enable/disable landmark smoothing
 * @param enabled true to enable smoothing, false to disable
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetSmoothingEnabled(
        JNIEnv* env,
        jobject /* this */,
        jboolean enabled) {
    
    if (g_faceLandmarker) {
        g_faceLandmarker->setSmoothingEnabled(enabled);
        LOGI("Landmark smoothing %s", enabled ? "enabled" : "disabled");
    }
}

/**
 * Reset landmark smoothing filter state
 * Call this when face is lost or tracking needs to be reset
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeResetSmoothing(
        JNIEnv* env,
        jobject /* this */) {
    
    if (g_faceLandmarker) {
        g_faceLandmarker->resetSmoothing();
        LOGI("Landmark smoothing filter reset");
    }
    if (g_faceDetector) {
        g_faceDetector->resetSmoothing();
        LOGI("Detection smoothing filter reset");
    }
}

/**
 * Set camera FOV for accurate head pose estimation
 * @param fovDegrees Vertical field of view in degrees
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetCameraFov(
        JNIEnv* env,
        jobject /* this */,
        jfloat fovDegrees) {

    FaceLandmarker::setCameraFov(fovDegrees);
    LOGI("Camera FOV set to %.1f degrees", fovDegrees);
}

/**
 * Get current camera FOV
 * @return Current vertical FOV in degrees
 */
extern "C" JNIEXPORT jfloat JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeGetCameraFov(
        JNIEnv* env,
        jobject /* this */) {

    return FaceLandmarker::getCameraFov();
}

// ============================================================================
// Filter Configuration JNI Methods
// ============================================================================

/**
 * Set landmark filter type
 * @param filterType 0=ONE_EURO, 1=TIME_BASED_EMA, 2=GAUSSIAN, 3=EMA, 4=NONE
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetLandmarkFilterType(
        JNIEnv* env,
        jobject /* this */,
        jint filterType) {

    if (g_faceLandmarker) {
        auto type = static_cast<mnncv::SmoothingFilterType>(filterType);
        g_faceLandmarker->setFilterType(type);
        LOGI("Landmark filter type set to %d", filterType);
    }
}

/**
 * Get current landmark filter type
 * @return Filter type: 0=ONE_EURO, 1=TIME_BASED_EMA, 2=GAUSSIAN, 3=EMA, 4=NONE
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeGetLandmarkFilterType(
        JNIEnv* env,
        jobject /* this */) {

    if (g_faceLandmarker) {
        return static_cast<jint>(g_faceLandmarker->getFilterType());
    }
    return 4; // NONE
}

/**
 * Configure landmark filter with detailed parameters
 * @param filterType Filter type (0-4)
 * @param oneEuroMinCutoff One Euro min_cutoff parameter
 * @param oneEuroBeta One Euro beta parameter
 * @param oneEuroDerivCutoff One Euro derivate_cutoff parameter
 * @param timeEmaTau Time-Based EMA tau parameter (seconds)
 * @param gaussianWindowSize Gaussian window size
 * @param gaussianSigma Gaussian sigma
 * @param emaAlpha Simple EMA alpha
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeConfigureLandmarkFilter(
        JNIEnv* env,
        jobject /* this */,
        jint filterType,
        jfloat oneEuroMinCutoff,
        jfloat oneEuroBeta,
        jfloat oneEuroDerivCutoff,
        jfloat timeEmaTau,
        jint gaussianWindowSize,
        jfloat gaussianSigma,
        jfloat emaAlpha) {

    if (g_faceLandmarker) {
        mnncv::FilterConfig config;
        config.type = static_cast<mnncv::SmoothingFilterType>(filterType);
        config.one_euro_min_cutoff = oneEuroMinCutoff;
        config.one_euro_beta = oneEuroBeta;
        config.one_euro_derivate_cutoff = oneEuroDerivCutoff;
        config.time_ema_tau = timeEmaTau;
        config.gaussian_window_size = gaussianWindowSize;
        config.gaussian_sigma = gaussianSigma;
        config.ema_alpha = emaAlpha;

        g_faceLandmarker->setFilterConfig(config);
        LOGI("Landmark filter configured: type=%d, tau=%.3f", filterType, timeEmaTau);
    }
}

/**
 * Set detection filter type
 * @param filterType 0=ONE_EURO, 1=TIME_BASED_EMA, 2=GAUSSIAN, 3=EMA, 4=NONE
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetDetectionFilterType(
        JNIEnv* env,
        jobject /* this */,
        jint filterType) {

    if (g_faceDetector) {
        auto type = static_cast<mnncv::SmoothingFilterType>(filterType);
        g_faceDetector->setFilterType(type);
        LOGI("Detection filter type set to %d", filterType);
    }
}

/**
 * Get current detection filter type
 * @return Filter type: 0=ONE_EURO, 1=TIME_BASED_EMA, 2=GAUSSIAN, 3=EMA, 4=NONE
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeGetDetectionFilterType(
        JNIEnv* env,
        jobject /* this */) {

    if (g_faceDetector) {
        return static_cast<jint>(g_faceDetector->getFilterType());
    }
    return 4; // NONE
}

/**
 * Configure detection filter with detailed parameters
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeConfigureDetectionFilter(
        JNIEnv* env,
        jobject /* this */,
        jint filterType,
        jfloat oneEuroMinCutoff,
        jfloat oneEuroBeta,
        jfloat oneEuroDerivCutoff,
        jfloat timeEmaTau,
        jint gaussianWindowSize,
        jfloat gaussianSigma,
        jfloat emaAlpha) {

    if (g_faceDetector) {
        mnncv::FilterConfig config;
        config.type = static_cast<mnncv::SmoothingFilterType>(filterType);
        config.one_euro_min_cutoff = oneEuroMinCutoff;
        config.one_euro_beta = oneEuroBeta;
        config.one_euro_derivate_cutoff = oneEuroDerivCutoff;
        config.time_ema_tau = timeEmaTau;
        config.gaussian_window_size = gaussianWindowSize;
        config.gaussian_sigma = gaussianSigma;
        config.ema_alpha = emaAlpha;

        g_faceDetector->setFilterConfig(config);
        LOGI("Detection filter configured: type=%d, tau=%.3f", filterType, timeEmaTau);
    }
}

/**
 * Set head pose filter type (for estimateHeadPose smoothing)
 * @param filterType 0=ONE_EURO, 1=GAUSSIAN, 2=EMA, 3=TIME_BASED_EMA
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeSetHeadPoseFilterType(
        JNIEnv* env,
        jobject /* this */,
        jint filterType) {

    FaceLandmarker::setHeadPoseFilterTypeInt(filterType);
    LOGI("Head pose filter type set to %d", filterType);
}

// ============================================================================
// YUV Direct Input Support - Zero-copy pipeline optimization
// ============================================================================

/**
 * Detect face directly from YUV data with rotation
 * This integrates YUV->RGB conversion into the preprocessing pipeline
 * 
 * @param yBuffer Y plane ByteBuffer (direct)
 * @param uBuffer U plane ByteBuffer (direct)
 * @param vBuffer V plane ByteBuffer (direct)
 * @param yRowStride Y plane row stride
 * @param uvRowStride UV plane row stride
 * @param uvPixelStride UV pixel stride (1=I420, 2=NV12/NV21)
 * @param width Source image width
 * @param height Source image height
 * @param rotation Rotation in degrees (0, 90, 180, 270)
 * @param scoreThreshold Detection score threshold
 * @param iouThreshold NMS IOU threshold
 * @return Float array [x1, y1, x2, y2, score] or null if no face
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectFaceFromYuv(
        JNIEnv* env,
        jobject /* this */,
        jobject yBuffer,
        jobject uBuffer,
        jobject vBuffer,
        jint yRowStride,
        jint uvRowStride,
        jint uvPixelStride,
        jint width,
        jint height,
        jint rotation,
        jfloat scoreThreshold,
        jfloat iouThreshold) {
    
    if (!g_faceDetector) {
        LOGE("Face detector not initialized");
        return nullptr;
    }
    
    // Get direct buffer addresses
    auto yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto uData = static_cast<uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto vData = static_cast<uint8_t*>(env->GetDirectBufferAddress(vBuffer));
    
    if (!yData || !uData || !vData) {
        LOGE("Failed to get direct buffer addresses");
        return nullptr;
    }
    
    // Detect faces directly from YUV
    auto faces = g_faceDetector->detectFromYuv(
        yData, uData, vData,
        yRowStride, uvRowStride, uvPixelStride,
        width, height, rotation,
        scoreThreshold, iouThreshold);
    
    if (faces.empty()) {
        return nullptr;
    }
    
    // Return first face as [x1, y1, x2, y2, score]
    FaceDetection& face = faces[0];
    jfloatArray result = env->NewFloatArray(5);
    float boxData[5] = { face.x1, face.y1, face.x2, face.y2, face.score };
    env->SetFloatArrayRegion(result, 0, 5, boxData);
    
    return result;
}

/**
 * Detect landmarks directly from YUV data with pre-computed bounding box
 * 
 * @param yBuffer Y plane ByteBuffer (direct)
 * @param uBuffer U plane ByteBuffer (direct)
 * @param vBuffer V plane ByteBuffer (direct)
 * @param yRowStride Y plane row stride
 * @param uvRowStride UV plane row stride
 * @param uvPixelStride UV pixel stride
 * @param width Image width (after rotation)
 * @param height Image height (after rotation)
 * @param rotation Rotation in degrees
 * @param boxX1, boxY1, boxX2, boxY2 Face bounding box (normalized, in rotated coords)
 * @param presenceThreshold Presence threshold
 * @return Float array of landmarks [x,y,z, ..., presence_score] or null on failure
 *         The last element is presence_score
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectLandmarksFromYuv(
        JNIEnv* env,
        jobject /* this */,
        jobject yBuffer,
        jobject uBuffer,
        jobject vBuffer,
        jint yRowStride,
        jint uvRowStride,
        jint uvPixelStride,
        jint width,
        jint height,
        jint rotation,
        jfloat boxX1,
        jfloat boxY1,
        jfloat boxX2,
        jfloat boxY2,
        jfloat presenceThreshold) {
    
    if (!g_faceLandmarker) {
        LOGE("Face landmarker not initialized");
        return nullptr;
    }
    
    auto yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto uData = static_cast<uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto vData = static_cast<uint8_t*>(env->GetDirectBufferAddress(vBuffer));
    
    if (!yData || !uData || !vData) {
        LOGE("Failed to get direct buffer addresses");
        return nullptr;
    }
    
    // Create FaceDetection from provided box
    FaceDetection face;
    face.x1 = boxX1;
    face.y1 = boxY1;
    face.x2 = boxX2;
    face.y2 = boxY2;
    face.score = 1.0f;
    
    // Run landmark detection directly from YUV
    auto result = g_faceLandmarker->detectFromYuv(
        yData, uData, vData,
        yRowStride, uvRowStride, uvPixelStride,
        width, height, rotation,
        face, presenceThreshold);
    
    if (!result.isValid()) {
        return nullptr;
    }
    
    // Convert landmarks to float array with presence_score appended
    size_t numLandmarks = result.landmarks.size();
    jfloatArray landmarksArray = env->NewFloatArray(numLandmarks * 3 + 1);
    
    std::vector<float> landmarksData;
    landmarksData.reserve(numLandmarks * 3 + 1);
    
    for (const auto& lm : result.landmarks) {
        landmarksData.push_back(lm.x);
        landmarksData.push_back(lm.y);
        landmarksData.push_back(lm.z);
    }
    
    // Append presence_score as the last element
    landmarksData.push_back(result.presence_score);
    
    env->SetFloatArrayRegion(landmarksArray, 0, numLandmarks * 3 + 1, landmarksData.data());
    
    return landmarksArray;
}

/**
 * Full pipeline: Detect face and landmarks from YUV in one call
 * Combines face detection and landmark detection using YUV input
 * 
 * @return Float array of landmarks [x,y,z, ...] or null if no face
 */
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_mediapipefacelandmark_MediaPipeFaceDetector_nativeDetectAllFromYuv(
        JNIEnv* env,
        jobject /* this */,
        jobject yBuffer,
        jobject uBuffer,
        jobject vBuffer,
        jint yRowStride,
        jint uvRowStride,
        jint uvPixelStride,
        jint width,
        jint height,
        jint rotation,
        jfloat scoreThreshold,
        jfloat iouThreshold,
        jfloat presenceThreshold) {
    
    if (!g_faceDetector || !g_faceLandmarker) {
        LOGE("Models not initialized");
        return nullptr;
    }
    
    auto yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto uData = static_cast<uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto vData = static_cast<uint8_t*>(env->GetDirectBufferAddress(vBuffer));
    
    if (!yData || !uData || !vData) {
        LOGE("Failed to get direct buffer addresses");
        return nullptr;
    }
    
    // Step 1: Detect face from YUV
    auto faces = g_faceDetector->detectFromYuv(
        yData, uData, vData,
        yRowStride, uvRowStride, uvPixelStride,
        width, height, rotation,
        scoreThreshold, iouThreshold);
    
    if (faces.empty()) {
        return nullptr;
    }
    
    // Step 2: Detect landmarks from YUV using the detected face
    auto result = g_faceLandmarker->detectFromYuv(
        yData, uData, vData,
        yRowStride, uvRowStride, uvPixelStride,
        width, height, rotation,
        faces[0], presenceThreshold);
    
    if (!result.isValid()) {
        return nullptr;
    }
    
    // Convert landmarks to float array
    size_t numLandmarks = result.landmarks.size();
    jfloatArray landmarksArray = env->NewFloatArray(numLandmarks * 3);
    
    std::vector<float> landmarksData;
    landmarksData.reserve(numLandmarks * 3);
    
    for (const auto& lm : result.landmarks) {
        landmarksData.push_back(lm.x);
        landmarksData.push_back(lm.y);
        landmarksData.push_back(lm.z);
    }
    
    env->SetFloatArrayRegion(landmarksArray, 0, numLandmarks * 3, landmarksData.data());
    
    return landmarksArray;
}

//=============================================================================
// NEON Optimization Test JNI Methods
//=============================================================================

#include "face_geomentry/test_neon_optimization.h"

/**
 * Run NEON optimization tests and benchmarks
 * @return true if all tests passed
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mediapipefacelandmark_NeonOptimizationTest_runTests(
        JNIEnv* env,
        jclass /* clazz */) {
    
    LOGI("Running NEON Optimization Tests...");
    
    try {
        MNN::FaceGeometry::runNeonTests();
        LOGI("NEON Optimization Tests completed successfully");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("NEON Optimization Tests failed: %s", e.what());
        return JNI_FALSE;
    } catch (...) {
        LOGE("NEON Optimization Tests failed with unknown exception");
        return JNI_FALSE;
    }
}

/**
 * Check if NEON is enabled on this device
 * @return true if NEON is available
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mediapipefacelandmark_NeonOptimizationTest_isNeonEnabled(
        JNIEnv* env,
        jclass /* clazz */) {
    
#if MNN_USE_NEON
    LOGI("NEON is ENABLED");
    return JNI_TRUE;
#else
    LOGI("NEON is DISABLED");
    return JNI_FALSE;
#endif
}

/**
 * Run only benchmarks (skip correctness tests)
 */
extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_NeonOptimizationTest_runBenchmarksOnly(
        JNIEnv* env,
        jclass /* clazz */) {
    
    LOGI("Running NEON Benchmarks Only...");
    
    try {
        srand(42);  // Fixed seed for reproducibility
        MNN::FaceGeometry::NeonTestRunner::printHeader("Performance Benchmarks (Direct)");
        MNN::FaceGeometry::NeonTestRunner::runBenchmarks();
        LOGI("NEON Benchmarks completed");
    } catch (const std::exception& e) {
        LOGE("NEON Benchmarks failed: %s", e.what());
    }
}
