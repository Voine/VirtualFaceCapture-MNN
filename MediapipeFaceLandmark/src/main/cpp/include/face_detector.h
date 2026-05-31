/**
 * @file face_detector.h
 * @brief Face Detection using MediaPipe BlazeFace model with MNN
 * 
 * Detects faces in images with bounding boxes and 6 keypoints.
 * Follows MediaPipe's anchor generation and box decoding strategy.
 */

#ifndef MEDIAPIPE_FACE_DETECTOR_H
#define MEDIAPIPE_FACE_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include "common/types.h"
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/Tensor.hpp"

// Include filter header for complete types
#include "face_pipeline_filter.h"

namespace mediapipe_face {

/**
 * @brief Face Detector using MediaPipe BlazeFace short-range model
 * 
 * This class handles:
 * - Loading and running the face detection MNN model
 * - SSD anchor generation (MediaPipe style)
 * - Box decoding and NMS
 * - Optional detection smoothing
 * - YUV direct input support
 */
class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();
    
    // Prevent copying
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    /**
     * @brief Load the face detection model
     * @param mnn_path Path to the MNN model file
     * @param num_threads Number of threads for inference
     * @return true if successful
     */
    bool loadModel(const std::string& mnn_path, int num_threads = 2);

    /**
     * @brief Detect faces from RGB image data
     * @param img_data RGB image data (uint8, HWC format)
     * @param width Image width
     * @param height Image height
     * @param score_threshold Detection score threshold
     * @param iou_threshold NMS IOU threshold
     * @return Vector of detected faces with normalized coordinates [0,1]
     */
    std::vector<FaceDetection> detect(const uint8_t* img_data, int width, int height,
                                       float score_threshold = 0.5f,
                                       float iou_threshold = 0.3f);

    /**
     * @brief Detect faces directly from YUV data (zero-copy optimization)
     * @param yData Y plane data
     * @param uData U plane data
     * @param vData V plane data
     * @param yRowStride Y plane row stride
     * @param uvRowStride UV plane row stride
     * @param uvPixelStride UV pixel stride (1=I420, 2=NV12/NV21)
     * @param width Source image width
     * @param height Source image height
     * @param rotation Rotation in degrees (0, 90, 180, 270)
     * @param score_threshold Detection score threshold
     * @param iou_threshold NMS IOU threshold
     * @return Vector of detected faces in rotated image coordinates
     */
    std::vector<FaceDetection> detectFromYuv(
        const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
        int yRowStride, int uvRowStride, int uvPixelStride,
        int width, int height, int rotation,
        float score_threshold = 0.5f, float iou_threshold = 0.3f);

    // ========== Smoothing Filter Configuration ==========
    
    void setSmoothingEnabled(bool enabled) { smoothing_enabled_ = enabled; }
    [[nodiscard]] bool isSmoothingEnabled() const { return smoothing_enabled_; }
    void resetSmoothing();
    
    void setFilterType(mnncv::SmoothingFilterType type);
    void setFilterConfig(const mnncv::FilterConfig& config);
    [[nodiscard]] mnncv::SmoothingFilterType getFilterType() const;

    // ========== Model Info ==========
    
    void getInputSize(int& height, int& width) const {
        height = input_h_;
        width = input_w_;
    }
    
    [[nodiscard]] std::string lastError() const { return last_error_; }

    // ========== Debug ==========
    
    static void setDumpDirectory(const std::string& dir);

private:
    // SSD anchor generation (MediaPipe style)
    void generateAnchors();

    // Box decoding from raw model outputs
    void decodeBoxes(const float* raw_boxes, const float* raw_scores,
                     std::vector<FaceDetection>& detections,
                     float score_threshold);

    // Non-maximum suppression (weighted NMS like MediaPipe)
    static void nms(std::vector<FaceDetection>& input,
             std::vector<FaceDetection>& output,
             float iou_threshold);

    static float iou(const FaceDetection& a, const FaceDetection& b);

    // MNN components
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    // Model dimensions
    int input_h_ = 128;
    int input_w_ = 128;
    int input_c_ = 3;

    // SSD anchors
    std::vector<Anchor> anchors_;

    // MediaPipe face detection parameters
    static constexpr int kNumCoords = 16;              // 4 box + 6 keypoints * 2
    static constexpr int kNumKeypoints = 6;
    static constexpr int kBoxCoordOffset = 0;
    static constexpr int kKeypointCoordOffset = 4;
    static constexpr int kNumValuesPerKeypoint = 2;
    static constexpr float kXScale = 128.0f;
    static constexpr float kYScale = 128.0f;
    static constexpr float kWScale = 128.0f;
    static constexpr float kHScale = 128.0f;
    static constexpr float kScoreClippingThresh = 100.0f;

    std::string last_error_;
    
    // Smoothing filter
    bool smoothing_enabled_ = false;
    std::unique_ptr<mnncv::UnifiedDetectionSmoothingFilter> detection_filter_;
};

} // namespace mediapipe_face

#endif // MEDIAPIPE_FACE_DETECTOR_H
