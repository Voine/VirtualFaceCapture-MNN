/**
 * @file face_landmarker.h
 * @brief Face Landmark Detection using MediaPipe model with MNN
 * 
 * Detects 478 facial landmarks (468 face mesh + 10 iris landmarks).
 * Includes head pose estimation using Face Geometry pipeline.
 */

#ifndef MEDIAPIPE_FACE_LANDMARKER_H
#define MEDIAPIPE_FACE_LANDMARKER_H

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/Tensor.hpp"

// Include filter header for complete types (defines mnncv namespace)
#include "face_pipeline_filter.h"

// Forward declarations

namespace MNN::FaceGeometry {
class GeometryPipeline;
struct FaceGeometry;
}


namespace mediapipe_face {

/**
 * @brief Face Landmarker using MediaPipe Face Mesh model
 * 
 * This class handles:
 * - Loading and running the landmark detection MNN model
 * - Face region transformation (crop, rotate, resize)
 * - Landmark decoding and coordinate projection
 * - Head pose estimation using Face Geometry pipeline
 * - Optional landmark smoothing
 * - YUV direct input support
 */
class FaceLandmarker {
public:
    FaceLandmarker();
    ~FaceLandmarker();
    
    // Prevent copying
    FaceLandmarker(const FaceLandmarker&) = delete;
    FaceLandmarker& operator=(const FaceLandmarker&) = delete;

    /**
     * @brief Load the landmark detection model
     * @param mnn_path Path to the MNN model file
     * @param num_threads Number of threads for inference
     * @return true if successful
     */
    bool loadModel(const std::string& mnn_path, int num_threads = 2);

    /**
     * @brief Detect landmarks from RGB image data
     * @param img_data RGB image data (uint8, HWC format)
     * @param img_width Image width
     * @param img_height Image height
     * @param face_rect Face detection result for cropping
     * @param presence_threshold Landmark presence threshold
     * @return Landmarks result with normalized coordinates [0,1]
     */
    FaceLandmarksResult detect(const uint8_t* img_data, int img_width, int img_height,
                                const FaceDetection& face_rect,
                                float presence_threshold = 0.5f);

    /**
     * @brief Detect landmarks directly from YUV data
     * @param yData Y plane data
     * @param uData U plane data
     * @param vData V plane data
     * @param yRowStride Y plane row stride
     * @param uvRowStride UV plane row stride
     * @param uvPixelStride UV pixel stride
     * @param img_width Image width (after rotation)
     * @param img_height Image height (after rotation)
     * @param rotation Image rotation in degrees
     * @param face_rect Face detection result (in rotated image coords)
     * @param presence_threshold Landmark presence threshold
     * @return Landmarks result
     */
    FaceLandmarksResult detectFromYuv(
        const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
        int yRowStride, int uvRowStride, int uvPixelStride,
        int img_width, int img_height, int rotation,
        const FaceDetection& face_rect,
        float presence_threshold = 0.5f);

    // ========== Smoothing Filter Configuration ==========
    
    void setSmoothingEnabled(bool enabled) { smoothing_enabled_ = enabled; }
    bool isSmoothingEnabled() const { return smoothing_enabled_; }
    void resetSmoothing();

    [[maybe_unused]] void setSmoothingParams(float min_cutoff, float beta, float derivate_cutoff);
    void setFilterType(mnncv::SmoothingFilterType type);
    void setFilterConfig(const mnncv::FilterConfig& config);
    [[nodiscard]] mnncv::SmoothingFilterType getFilterType() const;

    // ========== Head Pose Estimation ==========
    
    /**
     * @brief Estimate head pose from facial landmarks
     * Uses MediaPipe's Face Geometry pipeline (Procrustes-based)
     */
    static HeadPose estimateHeadPose(const std::vector<FaceLandmark>& landmarks,
                                      int image_width, int image_height);
    
    /**
     * @brief Get full face geometry result including 3D metric landmarks
     */
    static bool estimateFaceGeometry(const std::vector<FaceLandmark>& landmarks,
                                      int image_width, int image_height,
                                      MNN::FaceGeometry::FaceGeometry& result);
    
    // Head pose filter configuration
    [[maybe_unused]] static void setHeadPoseSmoothingEnabled(bool enabled);

    [[maybe_unused]] static bool isHeadPoseSmoothingEnabled();

    [[maybe_unused]] static void resetHeadPoseSmoothing();
    static void setHeadPoseFilterType(mnncv::HeadPoseFilterType type);

    [[maybe_unused]] static mnncv::HeadPoseFilterType getHeadPoseFilterType();
    static void setHeadPoseFilterTypeInt(int type);
    
    // Camera FOV configuration for accurate head pose
    static void setCameraFov(float fovDegrees);
    static float getCameraFov();

    // ========== Model Info ==========
    
    void getInputSize(int& height, int& width) const {
        height = input_h_;
        width = input_w_;
    }
    
    [[nodiscard]] std::string lastError() const { return last_error_; }

private:
    // Transform face detection rect to landmarker input rect
    static void transformFaceRect(const FaceDetection& detection,
                           float& center_x, float& center_y,
                           float& width, float& height, float& rotation);

    // Preprocess image for landmark detection
    void preprocessImage(const uint8_t* img_data, int img_width, int img_height,
                         float center_x, float center_y, float width, float height,
                         float rotation, float* output);

    // Preprocess YUV image directly
    void preprocessImageFromYuv(const uint8_t* yuvData, MNN::CV::ImageFormat yuvFormat,
                                int img_width, int img_height, int img_rotation,
                                float center_x, float center_y, float width, float height,
                                float face_rotation, float* output);

    // Decode model outputs to landmarks
    void decodeLandmarks(const float* landmarks_tensor,
                         float center_x, float center_y,
                         float rect_width, float rect_height,
                         float rotation,
                         FaceLandmarksResult& result) const;

    // Initialize geometry pipeline (lazy initialization)
    static bool initGeometryPipeline();

    [[maybe_unused]] static MNN::FaceGeometry::GeometryPipeline* getGeometryPipeline();

    // MNN components
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    // Model dimensions
    int input_h_ = 192;
    int input_w_ = 192;
    int input_c_ = 3;

    // Number of landmarks (468 mesh + 10 iris)
    static constexpr int kNumLandmarks = 478;

    std::string last_error_;
    
    // Landmark smoothing filter
    bool smoothing_enabled_ = true;
    std::unique_ptr<mnncv::UnifiedLandmarksSmoothingFilter> landmarks_filter_;

    // Static geometry pipeline (shared across all instances)
    static std::unique_ptr<MNN::FaceGeometry::GeometryPipeline> s_geometry_pipeline_;
    static bool s_geometry_initialized_;
    static float s_camera_fov_degrees_;
    
    // Static head pose filter
    static bool s_head_pose_filter_enabled_;
    static mnncv::HeadPoseFilterType s_head_pose_filter_type_;
};

} // namespace mediapipe_face

#endif // MEDIAPIPE_FACE_LANDMARKER_H
