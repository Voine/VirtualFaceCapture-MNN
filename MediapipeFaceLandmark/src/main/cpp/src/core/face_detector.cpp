/**
 * @file face_detector.cpp
 * @brief Face Detection implementation using MediaPipe BlazeFace model
 * 
 * Migrated from simple_mediapipe_face.cpp (SimpleMediaPipeFace class)
 */

#include "face_detector.h"
#include "common/logging.h"
#include "common/perf_profiler.h"
#include "utils/yuv_utils.h"
#include "face_pipeline_filter.h"

#include <algorithm>
#include <cmath>
#include <fstream>

// Override log tag for this module
#undef FACE_LOG_TAG
#define FACE_LOG_TAG "FaceDetector"

// Use filter types from mnncv namespace
using mnncv::UnifiedDetectionSmoothingFilter;
using mnncv::SmoothingFilterType;
using mnncv::FilterConfig;

namespace mediapipe_face {

// Static variables for debug frame dump
static bool s_first_frame_dumped = false;
static std::string s_dump_directory;

// Performance profiler instance
static DetectionPerfProfiler s_perf_profiler;

// ============================================================================
// Helper Functions
// ============================================================================

static void dumpFrameToPPM(const uint8_t* rgb_data, int width, int height, 
                           const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        FACE_LOGI("Failed to open file for writing: %s", filepath.c_str());
        return;
    }
    
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb_data), width * height * 3);
    file.close();
    
    FACE_LOGI("Dumped frame to: %s (%dx%d)", filepath.c_str(), width, height);
}

// Helper function to convert YUV to RGB and dump (with rotation support)
static void dumpYuvFrameToPPM(const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
                               int yRowStride, int uvRowStride, int uvPixelStride,
                               int srcWidth, int srcHeight, int rotation,
                               const std::string& filepath) {
    // Calculate output dimensions after rotation
    int outWidth, outHeight;
    if (rotation == 90 || rotation == 270) {
        outWidth = srcHeight;
        outHeight = srcWidth;
    } else {
        outWidth = srcWidth;
        outHeight = srcHeight;
    }
    
    // Allocate RGB buffer for rotated output
    std::vector<uint8_t> rgbBuffer(outWidth * outHeight * 3);
    
    // Convert YUV to RGB with rotation
    for (int dstY = 0; dstY < outHeight; dstY++) {
        for (int dstX = 0; dstX < outWidth; dstX++) {
            // Calculate source coordinates based on rotation
            int srcX, srcY;
            switch (rotation) {
                case 90:
                    srcX = dstY;
                    srcY = outWidth - 1 - dstX;
                    break;
                case 180:
                    srcX = srcWidth - 1 - dstX;
                    srcY = srcHeight - 1 - dstY;
                    break;
                case 270:
                    srcX = outHeight - 1 - dstY;
                    srcY = dstX;
                    break;
                default: // 0
                    srcX = dstX;
                    srcY = dstY;
                    break;
            }
            
            // Get Y value
            int yIndex = srcY * yRowStride + srcX;
            int yVal = yData[yIndex];
            
            // Get UV values (subsampled by 2)
            int uvIndex = (srcY / 2) * uvRowStride + (srcX / 2) * uvPixelStride;
            int u = uData[uvIndex] - 128;
            int v = vData[uvIndex] - 128;
            
            // YUV to RGB conversion using integer math
            int r = yVal + ((351 * v) >> 8);
            int g = yVal - ((86 * u + 179 * v) >> 8);
            int b = yVal + ((443 * u) >> 8);
            
            // Clamp and write to output
            int outIndex = (dstY * outWidth + dstX) * 3;
            rgbBuffer[outIndex + 0] = static_cast<uint8_t>(std::max(0, std::min(255, r)));
            rgbBuffer[outIndex + 1] = static_cast<uint8_t>(std::max(0, std::min(255, g)));
            rgbBuffer[outIndex + 2] = static_cast<uint8_t>(std::max(0, std::min(255, b)));
        }
    }
    
    // Write PPM file
    dumpFrameToPPM(rgbBuffer.data(), outWidth, outHeight, filepath);
    FACE_LOGI("Dumped YUV frame with rotation=%d: %dx%d -> %dx%d", 
              rotation, srcWidth, srcHeight, outWidth, outHeight);
}

// ============================================================================
// FaceDetector Implementation
// ============================================================================

FaceDetector::FaceDetector() {
    // Initialize detection box smoothing filter
    detection_filter_ = std::make_unique<UnifiedDetectionSmoothingFilter>();
    
    FilterConfig config;
    config.type = SmoothingFilterType::NONE;
    detection_filter_->configure(config);
    
    FACE_LOGI("FaceDetector: Time-Based EMA filter initialized (tau=0.06s)");
}

FaceDetector::~FaceDetector() {
    if (interpreter_) {
        interpreter_->releaseModel();
        if (session_) {
            interpreter_->releaseSession(session_);
        }
    }
}

void FaceDetector::setDumpDirectory(const std::string& dir) {
    s_dump_directory = dir;
    s_first_frame_dumped = false;
    FACE_LOGI("Dump directory set to: %s", dir.c_str());
}

void FaceDetector::resetSmoothing() {
    if (detection_filter_) {
        detection_filter_->reset();
        FACE_LOGI("Detection smoothing filter reset");
    }
}

void FaceDetector::setFilterType(SmoothingFilterType type) {
    if (detection_filter_) {
        detection_filter_->setFilterType(type);
        FACE_LOGI("Detection filter type set to %d", static_cast<int>(type));
    }
}

void FaceDetector::setFilterConfig(const FilterConfig& config) {
    if (detection_filter_) {
        detection_filter_->configure(config);
        FACE_LOGI("Detection filter configured (type=%d)", static_cast<int>(config.type));
    }
}

SmoothingFilterType FaceDetector::getFilterType() const {
    return detection_filter_ ? detection_filter_->getFilterType() : SmoothingFilterType::NONE;
}

bool FaceDetector::loadModel(const std::string& mnn_path, int num_threads) {
    last_error_.clear();
    
    interpreter_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(mnn_path.c_str()));
    if (!interpreter_) {
        last_error_ = "createFromFile failed";
        return false;
    }
    
    MNN::ScheduleConfig config;
    config.numThread = num_threads;
    config.type = MNN_FORWARD_VULKAN;
    MNN::BackendConfig backendCfg;
    backendCfg.precision = MNN::BackendConfig::Precision_Low_BF16;
    config.backendConfig = &backendCfg;
    
    session_ = interpreter_->createSession(config);
    if (!session_) {
        last_error_ = "createSession failed";
        return false;
    }
    
    input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor_) {
        last_error_ = "getSessionInput failed";
        return false;
    }
    
    // Get actual input dimensions from model
    auto shape = input_tensor_->shape();
    if (shape.size() == 4) {
        FACE_LOGI("Input tensor shape: [%d, %d, %d, %d]",
                  shape[0], shape[1], shape[2], shape[3]);
        input_h_ = shape[1];
        input_w_ = shape[2];
        input_c_ = shape[3];
        if (input_c_ != 3) {
            // Might be NCHW
            input_c_ = shape[1];
            input_h_ = shape[2];
            input_w_ = shape[3];
        }
    }
    
    FACE_LOGI("Face detector model loaded: %dx%dx%d", input_w_, input_h_, input_c_);
    
    // Generate anchors for the model
    generateAnchors();
    
    return true;
}

void FaceDetector::generateAnchors() {
    // MediaPipe face detection short range SSD anchors configuration
    anchors_.clear();
    
    const int num_layers = 4;
    const float min_scale = 0.1484375f;
    const float max_scale = 0.75f;
    const int strides[] = {8, 16, 16, 16};
    const float anchor_offset_x = 0.5f;
    const float anchor_offset_y = 0.5f;
    
    int layer_id = 0;
    while (layer_id < num_layers) {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;
        
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < num_layers &&
               strides[last_same_stride_layer] == strides[layer_id]) {
            
            float scale = min_scale + (max_scale - min_scale) *
                          static_cast<float>(last_same_stride_layer) / (num_layers - 1.0f);
            
            aspect_ratios.push_back(1.0f);
            scales.push_back(scale);
            
            float scale_next;
            if (last_same_stride_layer == num_layers - 1) {
                scale_next = 1.0f;
            } else {
                scale_next = min_scale + (max_scale - min_scale) *
                             static_cast<float>(last_same_stride_layer + 1) / (num_layers - 1.0f);
            }
            scales.push_back(std::sqrt(scale * scale_next));
            aspect_ratios.push_back(1.0f);
            
            last_same_stride_layer++;
        }
        
        for (size_t i = 0; i < scales.size(); ++i) {
            float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }
        
        int stride = strides[layer_id];
        int feature_map_height = input_h_ / stride;
        int feature_map_width = input_w_ / stride;
        
        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (size_t a = 0; a < anchor_height.size(); ++a) {
                    float x_center = (static_cast<float>(x) + anchor_offset_x) / 
                                     static_cast<float>(feature_map_width);
                    float y_center = (static_cast<float>(y) + anchor_offset_y) / 
                                     static_cast<float>(feature_map_height);
                    
                    Anchor anchor{};
                    anchor.x_center = x_center;
                    anchor.y_center = y_center;
                    anchor.w = 1.0f;
                    anchor.h = 1.0f;
                    
                    anchors_.push_back(anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    
    FACE_LOGI("Generated %zu anchors for face detection", anchors_.size());
}

void FaceDetector::decodeBoxes(const float* raw_boxes, const float* raw_scores,
                                std::vector<FaceDetection>& detections,
                                float score_threshold) {
    detections.clear();
    const auto num_boxes = static_cast<int>(anchors_.size());
    
    for (int i = 0; i < num_boxes; ++i) {
        float score = raw_scores[i];
        
        // Clamp score to prevent overflow
        if (score < -kScoreClippingThresh) score = -kScoreClippingThresh;
        if (score > kScoreClippingThresh) score = kScoreClippingThresh;
        
        // Apply sigmoid
        score = 1.0f / (1.0f + std::exp(-score));
        
        if (score < score_threshold) continue;
        
        const int box_offset = i * kNumCoords + kBoxCoordOffset;
        
        float x_center = raw_boxes[box_offset + 0];
        float y_center = raw_boxes[box_offset + 1];
        float w = raw_boxes[box_offset + 2];
        float h = raw_boxes[box_offset + 3];
        
        const Anchor& anchor = anchors_[i];
        
        float x_center_decoded = x_center / kXScale * anchor.w + anchor.x_center;
        float y_center_decoded = y_center / kYScale * anchor.h + anchor.y_center;
        float w_decoded = w / kWScale * anchor.w;
        float h_decoded = h / kHScale * anchor.h;
        
        float ymin = std::max(0.0f, std::min(1.0f, y_center_decoded - h_decoded / 2.0f));
        float xmin = std::max(0.0f, std::min(1.0f, x_center_decoded - w_decoded / 2.0f));
        float ymax = std::max(0.0f, std::min(1.0f, y_center_decoded + h_decoded / 2.0f));
        float xmax = std::max(0.0f, std::min(1.0f, x_center_decoded + w_decoded / 2.0f));
        
        FaceDetection det;
        det.x1 = xmin;
        det.y1 = ymin;
        det.x2 = xmax;
        det.y2 = ymax;
        det.score = score;
        det.anchor_idx = i;
        
        // Decode keypoints
        for (int k = 0; k < kNumKeypoints; ++k) {
            const int kp_offset = box_offset + kKeypointCoordOffset + k * kNumValuesPerKeypoint;
            float kp_x = raw_boxes[kp_offset + 0];
            float kp_y = raw_boxes[kp_offset + 1];
            
            float kp_x_decoded = std::max(0.0f, std::min(1.0f,
                kp_x / kXScale * anchor.w + anchor.x_center));
            float kp_y_decoded = std::max(0.0f, std::min(1.0f,
                kp_y / kYScale * anchor.h + anchor.y_center));
            
            det.keypoints.emplace_back(kp_x_decoded, kp_y_decoded);
        }
        
        detections.push_back(det);
    }
}

float FaceDetector::iou(const FaceDetection& a, const FaceDetection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    return inter / (areaA + areaB - inter + 1e-6f);
}

void FaceDetector::nms(std::vector<FaceDetection>& input,
                        std::vector<FaceDetection>& output,
                        float iou_threshold) {
    if (input.empty()) return;
    
    // Sort by score descending
    std::sort(input.begin(), input.end(),
              [](const FaceDetection& a, const FaceDetection& b) {
                  return a.score > b.score;
              });
    
    std::vector<bool> suppressed(input.size(), false);
    
    // Weighted NMS (MediaPipe style)
    for (size_t i = 0; i < input.size(); ++i) {
        if (suppressed[i]) continue;
        
        FaceDetection weighted_det = input[i];
        float total_score = input[i].score;
        
        for (size_t j = i + 1; j < input.size(); ++j) {
            if (suppressed[j]) continue;
            
            float overlap = iou(input[i], input[j]);
            if (overlap > iou_threshold) {
                float weight = input[j].score;
                
                // Weighted average
                weighted_det.x1 = (weighted_det.x1 * total_score + input[j].x1 * weight) / 
                                  (total_score + weight);
                weighted_det.y1 = (weighted_det.y1 * total_score + input[j].y1 * weight) / 
                                  (total_score + weight);
                weighted_det.x2 = (weighted_det.x2 * total_score + input[j].x2 * weight) / 
                                  (total_score + weight);
                weighted_det.y2 = (weighted_det.y2 * total_score + input[j].y2 * weight) / 
                                  (total_score + weight);
                
                for (size_t k = 0; k < weighted_det.keypoints.size() && 
                     k < input[j].keypoints.size(); ++k) {
                    weighted_det.keypoints[k].first =
                        (weighted_det.keypoints[k].first * total_score + 
                         input[j].keypoints[k].first * weight) / (total_score + weight);
                    weighted_det.keypoints[k].second =
                        (weighted_det.keypoints[k].second * total_score + 
                         input[j].keypoints[k].second * weight) / (total_score + weight);
                }
                
                total_score += weight;
                suppressed[j] = true;
            }
        }
        
        output.push_back(weighted_det);
    }
}

std::vector<FaceDetection> FaceDetector::detect(const uint8_t* img_data,
                                                 int width, int height,
                                                 float score_threshold,
                                                 float iou_threshold) {
    PERF_START(total_timer);
    std::vector<FaceDetection> results;
    
    // Dump first frame for debugging
    if (!s_first_frame_dumped && !s_dump_directory.empty() && img_data != nullptr) {
        std::string filepath = s_dump_directory + "/detector_input_frame.ppm";
        dumpFrameToPPM(img_data, width, height, filepath);
        s_first_frame_dumped = true;
    }
    
    if (!interpreter_ || !session_ || !input_tensor_) {
        last_error_ = "Model not loaded";
        return results;
    }
    
    if (!img_data) {
        last_error_ = "Invalid image data";
        return results;
    }
    
    // ---- Preprocessing ----
    PERF_START(preprocess_timer);
    
    float scale = std::min(static_cast<float>(input_w_) / static_cast<float>(width),
                           static_cast<float>(input_h_) / static_cast<float>(height));
    float new_w = static_cast<float>(width) * scale;
    float new_h = static_cast<float>(height) * scale;
    float pad_w = (static_cast<float>(input_w_) - new_w) / 2.0f;
    float pad_h = (static_cast<float>(input_h_) - new_h) / 2.0f;
    
    std::shared_ptr<MNN::Tensor> input_nchw(
        new MNN::Tensor(input_tensor_, input_tensor_->getDimensionType()));
    
    MNN::CV::ImageProcess::Config norm_cfg;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    ::memcpy(norm_cfg.mean, mean_vals, sizeof(mean_vals));
    ::memcpy(norm_cfg.normal, norm_vals, sizeof(norm_vals));
    norm_cfg.sourceFormat = MNN::CV::RGB;
    norm_cfg.destFormat = MNN::CV::RGB;
    norm_cfg.filterType = MNN::CV::BILINEAR;
    norm_cfg.wrap = MNN::CV::ZERO;
    
    auto norm_proc = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(norm_cfg));
    
    MNN::CV::Matrix trans;
    trans.setIdentity();
    trans.postScale(scale, scale);
    trans.postTranslate(pad_w, pad_h);
    MNN::CV::Matrix inv;
    trans.invert(&inv);
    norm_proc->setMatrix(inv);
    norm_proc->convert(img_data, width, height, width * 3, input_nchw.get());
    
    input_tensor_->copyFromHostTensor(input_nchw.get());
    
    PERF_END(preprocess_timer, s_perf_profiler.preprocess);
    
    // ---- Inference ----
    PERF_START(inference_timer);
    interpreter_->runSession(session_);
    PERF_END(inference_timer, s_perf_profiler.inference);
    
    // ---- Postprocessing ----
    PERF_START(postprocess_timer);
    
    auto outputs = interpreter_->getSessionOutputAll(session_);
    
    MNN::Tensor* bbox_tensor = nullptr;
    MNN::Tensor* score_tensor = nullptr;
    
    for (auto& kv : outputs) {
        if (kv.first.find("regressor") != std::string::npos ||
            kv.first.find("boxes") != std::string::npos) {
            bbox_tensor = kv.second;
        } else if (kv.first.find("classificator") != std::string::npos ||
                   kv.first.find("scores") != std::string::npos) {
            score_tensor = kv.second;
        }
    }
    
    if (!bbox_tensor || !score_tensor) {
        last_error_ = "Could not find output tensors";
        return results;
    }
    
    MNN::Tensor host_bbox(bbox_tensor, bbox_tensor->getDimensionType());
    MNN::Tensor host_score(score_tensor, score_tensor->getDimensionType());
    bbox_tensor->copyToHostTensor(&host_bbox);
    score_tensor->copyToHostTensor(&host_score);
    
    const float* raw_boxes = host_bbox.host<float>();
    const float* raw_scores = host_score.host<float>();
    
    std::vector<FaceDetection> detections;
    decodeBoxes(raw_boxes, raw_scores, detections, score_threshold);
    nms(detections, results, iou_threshold);
    
    PERF_END(postprocess_timer, s_perf_profiler.postprocess);
    
    // ---- Coordinate Transform ----
    PERF_START(coord_timer);
    
    for (auto& det : results) {
        float x1_pixel = det.x1 * static_cast<float>(input_w_);
        float y1_pixel = det.y1 * static_cast<float>(input_h_);
        float x2_pixel = det.x2 * static_cast<float>(input_w_);
        float y2_pixel = det.y2 * static_cast<float>(input_h_);
        
        x1_pixel -= pad_w;
        y1_pixel -= pad_h;
        x2_pixel -= pad_w;
        y2_pixel -= pad_h;
        
        x1_pixel /= scale;
        y1_pixel /= scale;
        x2_pixel /= scale;
        y2_pixel /= scale;
        
        det.x1 = std::max(0.0f, std::min(1.0f, x1_pixel / static_cast<float>(width)));
        det.y1 = std::max(0.0f, std::min(1.0f, y1_pixel / static_cast<float>(height)));
        det.x2 = std::max(0.0f, std::min(1.0f, x2_pixel / static_cast<float>(width)));
        det.y2 = std::max(0.0f, std::min(1.0f, y2_pixel / static_cast<float>(height)));
        
        for (auto& kp : det.keypoints) {
            float kp_x_pixel = kp.first * static_cast<float>(input_w_);
            float kp_y_pixel = kp.second * static_cast<float>(input_h_);
            
            kp_x_pixel -= pad_w;
            kp_y_pixel -= pad_h;
            kp_x_pixel /= scale;
            kp_y_pixel /= scale;
            
            kp.first = std::max(0.0f, std::min(1.0f, kp_x_pixel / static_cast<float>(width)));
            kp.second = std::max(0.0f, std::min(1.0f, kp_y_pixel / static_cast<float>(height)));
        }
    }
    
    // Apply smoothing
    if (smoothing_enabled_ && detection_filter_ && results.size() == 1) {
        detection_filter_->apply(results[0].x1, results[0].y1,
                                  results[0].x2, results[0].y2);
    } else if (results.empty() && detection_filter_) {
        detection_filter_->reset();
    }
    
    PERF_END(coord_timer, s_perf_profiler.coord_transform);
    PERF_END(total_timer, s_perf_profiler.total);
    PERF_CHECK_LOG(s_perf_profiler, PERF_LOG_INTERVAL);
    
    return results;
}

std::vector<FaceDetection> FaceDetector::detectFromYuv(
    const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
    int yRowStride, int uvRowStride, int uvPixelStride,
    int width, int height, int rotation,
    float score_threshold, float iou_threshold) {
    
    PERF_START(total_timer);
    std::vector<FaceDetection> results;
    
    if (!interpreter_ || !session_ || !input_tensor_) {
        last_error_ = "Model not loaded";
        return results;
    }
    
    // Dump first frame for debugging (YUV input with rotation)
    if (!s_first_frame_dumped && !s_dump_directory.empty() && yData != nullptr) {
        std::string filepath = s_dump_directory + "/detector_input_frame_yuv.ppm";
        dumpYuvFrameToPPM(yData, uData, vData, yRowStride, uvRowStride, uvPixelStride,
                          width, height, rotation, filepath);
        s_first_frame_dumped = true;
    }
    
    // Prepare YUV buffer
    auto yuvFormat = yuv::detectFormat(uvPixelStride, uData, vData);
    std::vector<uint8_t> yuvBuffer;
    yuv::prepareBuffer(yData, uData, vData, yRowStride, uvRowStride, uvPixelStride,
                       width, height, yuvFormat, yuvBuffer);
    
    // Calculate output dimensions after rotation
    int srcWidth = width;
    int srcHeight = height;
    int dstWidth, dstHeight;
    yuv::getRotatedDimensions(srcWidth, srcHeight, rotation, dstWidth, dstHeight);
    
    // ---- Preprocessing with YUV input ----
    PERF_START(preprocess_timer);
    
    float scale = std::min(static_cast<float>(input_w_) / static_cast<float>(dstWidth),
                           static_cast<float>(input_h_) / static_cast<float>(dstHeight));
    float new_w = static_cast<float>(dstWidth) * scale;
    float new_h = static_cast<float>(dstHeight) * scale;
    float pad_w = (static_cast<float>(input_w_) - new_w) / 2.0f;
    float pad_h = (static_cast<float>(input_h_) - new_h) / 2.0f;
    
    std::shared_ptr<MNN::Tensor> input_nchw(
        new MNN::Tensor(input_tensor_, input_tensor_->getDimensionType()));
    
    // Configure MNN ImageProcess: YUV -> RGB with rotation + scale + normalize
    MNN::CV::ImageProcess::Config config;
    config.sourceFormat = yuvFormat;
    config.destFormat = MNN::CV::RGB;
    config.filterType = MNN::CV::BILINEAR;
    config.wrap = MNN::CV::ZERO;
    
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    ::memcpy(config.mean, mean_vals, sizeof(mean_vals));
    ::memcpy(config.normal, norm_vals, sizeof(norm_vals));
    
    auto process = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(config));
    
    // Build transformation matrix
    MNN::CV::Matrix trans;
    trans.setIdentity();
    trans.postScale(1.0f / scale, 1.0f / scale);
    trans.postTranslate(-pad_w / scale, -pad_h / scale);
    
    if (rotation != 0) {
        // Apply rotation transformation
        float cx_dst = static_cast<float>(dstWidth) / 2.0f;
        float cy_dst = static_cast<float>(dstHeight) / 2.0f;
        float cx_src = static_cast<float>(srcWidth) / 2.0f;
        float cy_src = static_cast<float>(srcHeight) / 2.0f;
        
        trans.postTranslate(-cx_dst, -cy_dst);
        trans.postRotate(-static_cast<float>(rotation));
        trans.postTranslate(cx_src, cy_src);
    }
    
    process->setMatrix(trans);
    process->convert(yuvBuffer.data(), srcWidth, srcHeight, 0, input_nchw.get());
    input_tensor_->copyFromHostTensor(input_nchw.get());
    
    PERF_END(preprocess_timer, s_perf_profiler.preprocess);
    
    // ---- Inference ----
    PERF_START(inference_timer);
    interpreter_->runSession(session_);
    PERF_END(inference_timer, s_perf_profiler.inference);
    
    // ---- Postprocessing ----
    PERF_START(postprocess_timer);
    
    auto outputs = interpreter_->getSessionOutputAll(session_);
    
    MNN::Tensor* bbox_tensor = nullptr;
    MNN::Tensor* score_tensor = nullptr;
    
    for (auto& kv : outputs) {
        if (kv.first.find("regressor") != std::string::npos ||
            kv.first.find("boxes") != std::string::npos) {
            bbox_tensor = kv.second;
        } else if (kv.first.find("classificator") != std::string::npos ||
                   kv.first.find("scores") != std::string::npos) {
            score_tensor = kv.second;
        }
    }
    
    if (!bbox_tensor || !score_tensor) {
        last_error_ = "Could not find output tensors";
        return results;
    }
    
    MNN::Tensor host_bbox(bbox_tensor, bbox_tensor->getDimensionType());
    MNN::Tensor host_score(score_tensor, score_tensor->getDimensionType());
    bbox_tensor->copyToHostTensor(&host_bbox);
    score_tensor->copyToHostTensor(&host_score);
    
    std::vector<FaceDetection> detections;
    decodeBoxes(host_bbox.host<float>(), host_score.host<float>(), 
                detections, score_threshold);
    nms(detections, results, iou_threshold);
    
    PERF_END(postprocess_timer, s_perf_profiler.postprocess);
    
    // ---- Coordinate Transform ----
    PERF_START(coord_timer);
    
    for (auto& det : results) {
        float x1_pixel = det.x1 * static_cast<float>(input_w_);
        float y1_pixel = det.y1 * static_cast<float>(input_h_);
        float x2_pixel = det.x2 * static_cast<float>(input_w_);
        float y2_pixel = det.y2 * static_cast<float>(input_h_);
        
        x1_pixel -= pad_w;
        y1_pixel -= pad_h;
        x2_pixel -= pad_w;
        y2_pixel -= pad_h;
        
        x1_pixel /= scale;
        y1_pixel /= scale;
        x2_pixel /= scale;
        y2_pixel /= scale;
        
        det.x1 = std::max(0.0f, std::min(1.0f, x1_pixel / static_cast<float>(dstWidth)));
        det.y1 = std::max(0.0f, std::min(1.0f, y1_pixel / static_cast<float>(dstHeight)));
        det.x2 = std::max(0.0f, std::min(1.0f, x2_pixel / static_cast<float>(dstWidth)));
        det.y2 = std::max(0.0f, std::min(1.0f, y2_pixel / static_cast<float>(dstHeight)));
        
        for (auto& kp : det.keypoints) {
            float kp_x_pixel = kp.first * static_cast<float>(input_w_);
            float kp_y_pixel = kp.second * static_cast<float>(input_h_);
            
            kp_x_pixel -= pad_w;
            kp_y_pixel -= pad_h;
            kp_x_pixel /= scale;
            kp_y_pixel /= scale;
            
            kp.first = std::max(0.0f, std::min(1.0f, kp_x_pixel / static_cast<float>(dstWidth)));
            kp.second = std::max(0.0f, std::min(1.0f, kp_y_pixel / static_cast<float>(dstHeight)));
        }
    }
    
    if (smoothing_enabled_ && detection_filter_ && results.size() == 1) {
        detection_filter_->apply(results[0].x1, results[0].y1,
                                  results[0].x2, results[0].y2);
    } else if (results.empty() && detection_filter_) {
        detection_filter_->reset();
    }
    
    PERF_END(coord_timer, s_perf_profiler.coord_transform);
    PERF_END(total_timer, s_perf_profiler.total);
    PERF_CHECK_LOG(s_perf_profiler, PERF_LOG_INTERVAL);
    
    return results;
}

} // namespace mediapipe_face
