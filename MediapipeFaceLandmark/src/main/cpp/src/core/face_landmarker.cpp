/**
 * @file face_landmarker.cpp
 * @brief Face Landmark Detection implementation using MediaPipe model
 * 
 * Migrated from simple_mediapipe_face.cpp (SimpleFaceLandmarker class)
 */

#include "face_landmarker.h"
#include "common/logging.h"
#include "common/perf_profiler.h"
#include "utils/yuv_utils.h"
#include "face_geomentry/geometry_pipeline_nodep.h"

#include <algorithm>
#include <cmath>

#undef FACE_LOG_TAG
#define FACE_LOG_TAG "FaceLandmarker"

// Use filter types from mnncv namespace
using mnncv::UnifiedLandmarksSmoothingFilter;
using mnncv::SmoothingFilterType;
using mnncv::FilterConfig;
using mnncv::OneEuroFilter3D;
using mnncv::GaussianMovingAverageFilter3D;
using mnncv::ExponentialMovingAverageFilter3D;
using mnncv::TimeBasedEMAFilter3D;

namespace mediapipe_face {

// Performance profiler
static LandmarkPerfProfiler s_perf_profiler;

// Static Member Initialization
std::unique_ptr<MNN::FaceGeometry::GeometryPipeline> FaceLandmarker::s_geometry_pipeline_ = nullptr;
bool FaceLandmarker::s_geometry_initialized_ = false;
float FaceLandmarker::s_camera_fov_degrees_ = 63.0f;
bool FaceLandmarker::s_head_pose_filter_enabled_ = true;
mnncv::HeadPoseFilterType FaceLandmarker::s_head_pose_filter_type_ = mnncv::HeadPoseFilterType::TIME_BASED_EMA;

// Static head pose filters
static std::unique_ptr<OneEuroFilter3D> s_head_pose_filter;
static std::unique_ptr<GaussianMovingAverageFilter3D> s_head_pose_gaussian_filter;
static std::unique_ptr<ExponentialMovingAverageFilter3D> s_head_pose_ema_filter;
static std::unique_ptr<TimeBasedEMAFilter3D> s_head_pose_time_ema_filter;

// Helper function for Euler angle extraction
// Uses ZYX intrinsic rotation (Tait-Bryan angles): R = Rz(roll) * Ry(yaw) * Rx(pitch)
// Only elements from column 0 (r00, r10, r20) and column 2's bottom (r21, r22) plus
// r11, r12 for gimbal lock case are needed for angle extraction.
namespace {
void extractEulerAnglesFromMatrix4f(const MNN::FaceGeometry::Matrix4f& mat,
                                    float& pitch, float& yaw, float& roll) {
    // Extract rotation matrix elements (only those needed for ZYX decomposition)
    float r00 = mat(0, 0);
    float r10 = mat(1, 0), r11 = mat(1, 1), r12 = mat(1, 2);
    float r20 = mat(2, 0), r21 = mat(2, 1), r22 = mat(2, 2);
    
    // Remove scale by normalizing the first column
    float scale = std::sqrt(r00*r00 + r10*r10 + r20*r20);
    if (scale > 1e-6f) {
        float inv_scale = 1.0f / scale;
        r00 *= inv_scale; r10 *= inv_scale; r20 *= inv_scale;
        r11 *= inv_scale; r12 *= inv_scale;
        r21 *= inv_scale; r22 *= inv_scale;
    }
    
    // Check for gimbal lock (when cos(yaw) ≈ 0)
    float sy = std::sqrt(r21 * r21 + r22 * r22);
    bool singular = sy < 1e-6f;
    
    float x_rot, y_rot, z_rot;
    if (!singular) {
        // Normal case: extract from row 3 and column 1
        x_rot = std::atan2(r21, r22);   // pitch from r21/r22
        y_rot = std::atan2(-r20, sy);    // yaw from r20
        z_rot = std::atan2(r10, r00);    // roll from r10/r00
    } else {
        // Gimbal lock: yaw ≈ ±90°, use alternative elements
        x_rot = std::atan2(-r12, r11);
        y_rot = std::atan2(-r20, sy);
        z_rot = 0.0f;
    }
    
    const float kRadToDeg = 180.0f / 3.14159265358979323846f;
    pitch = x_rot * kRadToDeg;
    yaw = y_rot * kRadToDeg;
    roll = z_rot * kRadToDeg;
}
} // anonymous namespace

// FaceLandmarker Implementation
FaceLandmarker::FaceLandmarker() {
    landmarks_filter_ = std::make_unique<UnifiedLandmarksSmoothingFilter>(kNumLandmarks);
    
    FilterConfig config;
    config.type = SmoothingFilterType::TIME_BASED_EMA;
//    config.time_ema_tau = 0.08f;  // 80 ms time
    landmarks_filter_->configure(config);
    
    FACE_LOGI("FaceLandmarker: Time-Based EMA Filter initialized (tau=%.3f)", config.time_ema_tau);
}

FaceLandmarker::~FaceLandmarker() {
    if (interpreter_) {
        interpreter_->releaseModel();
        if (session_) {
            interpreter_->releaseSession(session_);
        }
    }
}

bool FaceLandmarker::loadModel(const std::string& mnn_path, int num_threads) {
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
    
    auto shape = input_tensor_->shape();
    if (shape.size() == 4) {
        input_h_ = shape[1];
        input_w_ = shape[2];
        input_c_ = shape[3];
        if (input_c_ != 3) {
            input_c_ = shape[1];
            input_h_ = shape[2];
            input_w_ = shape[3];
        }
    }
    
    FACE_LOGI("Face landmarker model loaded: %dx%dx%d", input_w_, input_h_, input_c_);
    return true;
}

void FaceLandmarker::resetSmoothing() {
    if (landmarks_filter_) {
        landmarks_filter_->reset();
    }
}

[[maybe_unused]] void FaceLandmarker::setSmoothingParams(float min_cutoff, float beta, float derivate_cutoff) {
    if (landmarks_filter_) {
        FilterConfig config;
        config.type = SmoothingFilterType::ONE_EURO;
        config.one_euro_min_cutoff = min_cutoff;
        config.one_euro_beta = beta;
        config.one_euro_derivate_cutoff = derivate_cutoff;
        landmarks_filter_->configure(config);
    }
}

void FaceLandmarker::setFilterType(SmoothingFilterType type) {
    if (landmarks_filter_) {
        landmarks_filter_->setFilterType(type);
    }
}

void FaceLandmarker::setFilterConfig(const FilterConfig& config) {
    if (landmarks_filter_) {
        landmarks_filter_->configure(config);
    }
}

SmoothingFilterType FaceLandmarker::getFilterType() const {
    return landmarks_filter_ ? landmarks_filter_->getFilterType() : SmoothingFilterType::NONE;
}

void FaceLandmarker::transformFaceRect(const FaceDetection& detection,
                                        float& center_x, float& center_y,
                                        float& width, float& height,
                                        float& rotation) {
    rotation = 0.0f;
    if (detection.keypoints.size() >= 2) {
        float dx = detection.keypoints[1].first - detection.keypoints[0].first;
        float dy = detection.keypoints[1].second - detection.keypoints[0].second;
        rotation = std::atan2(dy, dx);
    }
    
    center_x = (detection.x1 + detection.x2) / 2.0f;
    center_y = (detection.y1 + detection.y2) / 2.0f;
    width = detection.x2 - detection.x1;
    height = detection.y2 - detection.y1;
    
    // MediaPipe RectTransformationCalculator: scale_x=1.5, scale_y=1.5, square_long=true
    width *= 1.5f;
    height *= 1.5f;
    float size = std::max(width, height);
    width = size;
    height = size;
}

void FaceLandmarker::preprocessImage(const uint8_t* img_data, int img_width, int img_height,
                                      float center_x, float center_y, float width, float height,
                                      float rotation, float* output) {
    const float cx_px = center_x * static_cast<float>(img_width);
    const float cy_px = center_y * static_cast<float>(img_height);
    const float w_px = width * static_cast<float>(img_width);
    const float h_px = height * static_cast<float>(img_height);
    
    MNN::CV::Matrix trans;
    trans.setIdentity();
    
    float scale_out = 1.0f / static_cast<float>(input_w_);
    trans.postScale(scale_out, scale_out);
    trans.postTranslate(0.5f * scale_out - 0.5f, 0.5f * scale_out - 0.5f);
    trans.postRotate(rotation * 180.0f / 3.14159265f);
    trans.postScale(w_px, h_px);
    trans.postTranslate(cx_px, cy_px);
    
    MNN::CV::ImageProcess::Config config;
    config.sourceFormat = MNN::CV::RGB;
    config.destFormat = MNN::CV::RGB;
    config.filterType = MNN::CV::BILINEAR;
    config.wrap = MNN::CV::ZERO;
    
    const float mean[4] = {127.5f, 127.5f, 127.5f, 0.0f};
    const float norm[4] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f, 0.0f};
    ::memcpy(config.mean, mean, sizeof(mean));
    ::memcpy(config.normal, norm, sizeof(norm));
    
    auto process = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(config));
    process->setMatrix(trans);
    
    std::vector<int> dims = {1, input_h_, input_w_, input_c_};
    std::shared_ptr<MNN::Tensor> temp_tensor(
        MNN::Tensor::create<float>(dims, nullptr, MNN::Tensor::TENSORFLOW));
    
    process->convert(img_data, img_width, img_height, img_width * 3, temp_tensor.get());
    ::memcpy(output, temp_tensor->host<float>(), input_h_ * input_w_ * 3 * sizeof(float));
}

void FaceLandmarker::preprocessImageFromYuv(const uint8_t* yuvData, MNN::CV::ImageFormat yuvFormat,
                                             int img_width, int img_height, int img_rotation,
                                             float center_x, float center_y, float width, float height,
                                             float face_rotation, float* output) {
    int srcWidth = (img_rotation == 90 || img_rotation == 270) ? img_height : img_width;
    int srcHeight = (img_rotation == 90 || img_rotation == 270) ? img_width : img_height;
    
    float cx_rotated = center_x * static_cast<float>(img_width);
    float cy_rotated = center_y * static_cast<float>(img_height);
    float w_rotated = width * static_cast<float>(img_width);
    float h_rotated = height * static_cast<float>(img_height);
    
    MNN::CV::Matrix trans;
    trans.setIdentity();
    
    float scale_out = 1.0f / static_cast<float>(input_w_);
    trans.postScale(scale_out, scale_out);
    trans.postTranslate(-0.5f, -0.5f);
    trans.postRotate(face_rotation * 180.0f / 3.14159265f);
    trans.postScale(w_rotated, h_rotated);
    trans.postTranslate(cx_rotated, cy_rotated);
    
    if (img_rotation != 0) {
        float cx_src = static_cast<float>(srcWidth) / 2.0f;
        float cy_src = static_cast<float>(srcHeight) / 2.0f;
        float cx_rot = static_cast<float>(img_width) / 2.0f;
        float cy_rot = static_cast<float>(img_height) / 2.0f;
        
        trans.postTranslate(-cx_rot, -cy_rot);
        trans.postRotate(-static_cast<float>(img_rotation));
        trans.postTranslate(cx_src, cy_src);
    }
    
    MNN::CV::ImageProcess::Config config;
    config.sourceFormat = yuvFormat;
    config.destFormat = MNN::CV::RGB;
    config.filterType = MNN::CV::BILINEAR;
    config.wrap = MNN::CV::ZERO;
    
    const float mean[4] = {127.5f, 127.5f, 127.5f, 0.0f};
    const float norm[4] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f, 0.0f};
    ::memcpy(config.mean, mean, sizeof(mean));
    ::memcpy(config.normal, norm, sizeof(norm));
    
    auto process = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(config));
    process->setMatrix(trans);
    
    std::vector<int> dims = {1, input_h_, input_w_, input_c_};
    std::shared_ptr<MNN::Tensor> temp_tensor(
        MNN::Tensor::create<float>(dims, nullptr, MNN::Tensor::TENSORFLOW));
    
    process->convert(yuvData, srcWidth, srcHeight, 0, temp_tensor.get());
    ::memcpy(output, temp_tensor->host<float>(), input_h_ * input_w_ * 3 * sizeof(float));
}

void FaceLandmarker::decodeLandmarks(const float* landmarks_tensor,
                                      float center_x, float center_y,
                                      float rect_width, float rect_height,
                                      float rotation,
                                      FaceLandmarksResult& result) const {
    const float cos_rot = std::cos(rotation);
    const float sin_rot = std::sin(rotation);
    result.landmarks.clear();
    result.landmarks.reserve(kNumLandmarks);
    
    for (int i = 0; i < kNumLandmarks; ++i) {
        float x = landmarks_tensor[i * 3 + 0];
        float y = landmarks_tensor[i * 3 + 1];
        float z = landmarks_tensor[i * 3 + 2];
        
        if (x > 2.0f || y > 2.0f) {
            x = x / static_cast<float>(input_w_);
            y = y / static_cast<float>(input_h_);
        }
        
        x = x - 0.5f;
        y = y - 0.5f;
        
        float rotated_x = x * cos_rot - y * sin_rot;
        float rotated_y = x * sin_rot + y * cos_rot;
        
        float final_x = center_x + rotated_x * rect_width;
        float final_y = center_y + rotated_y * rect_height;
        float final_z = z * rect_width;
        
        result.landmarks.push_back({final_x, final_y, final_z});
    }
}

FaceLandmarksResult FaceLandmarker::detect(const uint8_t* img_data, int img_width, int img_height,
                                            const FaceDetection& face_rect,
                                            float presence_threshold) {
    PERF_START(total_timer);
    FaceLandmarksResult result;
    result.presence = false;
    result.presence_score = 0.0f;
    
    if (!interpreter_ || !session_ || !input_tensor_) {
        last_error_ = "Model not loaded";
        return result;
    }
    
    PERF_START(preprocess_timer);
    
    float center_x, center_y, width, height, rotation;
    transformFaceRect(face_rect, center_x, center_y, width, height, rotation);
    
    std::vector<float> input_data(input_h_ * input_w_ * input_c_);
    preprocessImage(img_data, img_width, img_height,
                    center_x, center_y, width, height, rotation, input_data.data());
    
    auto input_host = new MNN::Tensor(input_tensor_, input_tensor_->getDimensionType());
    std::memcpy(input_host->host<float>(), input_data.data(), input_data.size() * sizeof(float));
    input_tensor_->copyFromHostTensor(input_host);
    delete input_host;
    
    PERF_END(preprocess_timer, s_perf_profiler.preprocess);
    
    PERF_START(inference_timer);
    interpreter_->runSession(session_);
    PERF_END(inference_timer, s_perf_profiler.inference);
    
    PERF_START(postprocess_timer);
    
    auto landmarks_tensor = interpreter_->getSessionOutput(session_, "Identity");
    auto face_flag_tensor = interpreter_->getSessionOutput(session_, "Identity_1");
    
    if (!landmarks_tensor || !face_flag_tensor) {
        last_error_ = "Failed to get output tensors";
        return result;
    }
    
    std::shared_ptr<MNN::Tensor> landmarks_host(
        new MNN::Tensor(landmarks_tensor, landmarks_tensor->getDimensionType()));
    landmarks_tensor->copyToHostTensor(landmarks_host.get());

    std::shared_ptr<MNN::Tensor> flag_host = std::make_shared<MNN::Tensor>(
            face_flag_tensor, MNN::Tensor::CAFFE);
    face_flag_tensor->copyToHostTensor(flag_host.get());
    float raw_score = flag_host->host<float>()[0];
    result.presence_score = 1.0f / (1.0f + std::exp(-raw_score));

    result.presence = result.presence_score >= presence_threshold;
    
    if (result.presence) {
        decodeLandmarks(landmarks_host->host<float>(), center_x, center_y, 
                        width, height, rotation, result);
    }
    
    PERF_END(postprocess_timer, s_perf_profiler.postprocess);
    
    PERF_START(smoothing_timer);
    
    if (result.presence && smoothing_enabled_ && landmarks_filter_ && !result.landmarks.empty()) {
        landmarks_filter_->apply(result.landmarks);
    } else if (!result.presence && landmarks_filter_) {
        landmarks_filter_->reset();
    }
    
    PERF_END(smoothing_timer, s_perf_profiler.smoothing);
    PERF_END(total_timer, s_perf_profiler.total);
    PERF_CHECK_LOG(s_perf_profiler, PERF_LOG_INTERVAL);
    
    return result;
}

FaceLandmarksResult FaceLandmarker::detectFromYuv(
    const uint8_t* yData, const uint8_t* uData, const uint8_t* vData,
    int yRowStride, int uvRowStride, int uvPixelStride,
    int img_width, int img_height, int rotation,
    const FaceDetection& face_rect,
    float presence_threshold) {
    
    PERF_START(total_timer);
    FaceLandmarksResult result;
    result.presence = false;
    result.presence_score = 0.0f;
    
    if (!interpreter_ || !session_ || !input_tensor_) {
        last_error_ = "Model not loaded";
        return result;
    }
    
    int srcWidth = (rotation == 90 || rotation == 270) ? img_height : img_width;
    int srcHeight = (rotation == 90 || rotation == 270) ? img_width : img_height;
    
    auto yuvFormat = yuv::detectFormat(uvPixelStride, uData, vData);
    std::vector<uint8_t> yuvBuffer;
    yuv::prepareBuffer(yData, uData, vData, yRowStride, uvRowStride, uvPixelStride,
                       srcWidth, srcHeight, yuvFormat, yuvBuffer);
    
    PERF_START(preprocess_timer);
    
    float center_x, center_y, width, height, face_rotation;
    transformFaceRect(face_rect, center_x, center_y, width, height, face_rotation);
    
    std::vector<float> input_data(input_h_ * input_w_ * input_c_);
    preprocessImageFromYuv(yuvBuffer.data(), yuvFormat,
                           img_width, img_height, rotation,
                           center_x, center_y, width, height, face_rotation,
                           input_data.data());
    
    auto input_host = new MNN::Tensor(input_tensor_, input_tensor_->getDimensionType());
    std::memcpy(input_host->host<float>(), input_data.data(), input_data.size() * sizeof(float));
    input_tensor_->copyFromHostTensor(input_host);
    delete input_host;
    
    PERF_END(preprocess_timer, s_perf_profiler.preprocess);
    
    PERF_START(inference_timer);
    interpreter_->runSession(session_);
    PERF_END(inference_timer, s_perf_profiler.inference);
    
    PERF_START(postprocess_timer);
    
    auto landmarks_tensor = interpreter_->getSessionOutput(session_, "Identity");
    auto face_flag_tensor = interpreter_->getSessionOutput(session_, "Identity_1");
    
    if (!landmarks_tensor || !face_flag_tensor) {
        last_error_ = "Failed to get output tensors";
        return result;
    }
    
    std::shared_ptr<MNN::Tensor> landmarks_host(
        new MNN::Tensor(landmarks_tensor, landmarks_tensor->getDimensionType()));
    landmarks_tensor->copyToHostTensor(landmarks_host.get());

    std::shared_ptr<MNN::Tensor> flag_host = std::make_shared<MNN::Tensor>(
            face_flag_tensor, MNN::Tensor::CAFFE);
    face_flag_tensor->copyToHostTensor(flag_host.get());
    float raw_score = flag_host->host<float>()[0];
    result.presence_score = 1.0f / (1.0f + std::exp(-raw_score));

    result.presence = result.presence_score >= presence_threshold;
    
    if (result.presence) {
        decodeLandmarks(landmarks_host->host<float>(), center_x, center_y,
                        width, height, face_rotation, result);
    }
    
    PERF_END(postprocess_timer, s_perf_profiler.postprocess);
    
    PERF_START(smoothing_timer);
    
    if (result.presence && smoothing_enabled_ && landmarks_filter_ && !result.landmarks.empty()) {
        landmarks_filter_->apply(result.landmarks);
    } else if (!result.presence && landmarks_filter_) {
        landmarks_filter_->reset();
    }
    
    PERF_END(smoothing_timer, s_perf_profiler.smoothing);
    PERF_END(total_timer, s_perf_profiler.total);
    PERF_CHECK_LOG(s_perf_profiler, PERF_LOG_INTERVAL);
    
    return result;
}

// Head Pose Estimation
[[maybe_unused]] void FaceLandmarker::setHeadPoseSmoothingEnabled(bool enabled) {
    s_head_pose_filter_enabled_ = enabled;
}

[[maybe_unused]] bool FaceLandmarker::isHeadPoseSmoothingEnabled() {
    return s_head_pose_filter_enabled_;
}

[[maybe_unused]] void FaceLandmarker::resetHeadPoseSmoothing() {
    if (s_head_pose_filter) s_head_pose_filter->reset();
    if (s_head_pose_gaussian_filter) s_head_pose_gaussian_filter->reset();
    if (s_head_pose_ema_filter) s_head_pose_ema_filter->reset();
    if (s_head_pose_time_ema_filter) s_head_pose_time_ema_filter->reset();
}

void FaceLandmarker::setHeadPoseFilterType(mnncv::HeadPoseFilterType type) {
    s_head_pose_filter_type_ = type;
    
    if (s_head_pose_filter) s_head_pose_filter->reset();
    if (s_head_pose_gaussian_filter) s_head_pose_gaussian_filter->reset();
    if (s_head_pose_ema_filter) s_head_pose_ema_filter->reset();
    if (s_head_pose_time_ema_filter) s_head_pose_time_ema_filter->reset();
    
    if (s_head_pose_filter_enabled_) {
        switch (type) {
            case mnncv::HeadPoseFilterType::ONE_EURO:
                if (!s_head_pose_filter) {
                    s_head_pose_filter = std::make_unique<OneEuroFilter3D>(
                            mnncv::DEFAULT_ONE_EURO_MIN_CUTOFF, mnncv::DEFAULT_ONE_EURO_BETA, mnncv::DEFAULT_ONE_EURO_DERIVATE_CUTOFF);
                }
                break;
            case mnncv::HeadPoseFilterType::GAUSSIAN_MOVING_AVG:
                if (!s_head_pose_gaussian_filter) {
                    s_head_pose_gaussian_filter = std::make_unique<GaussianMovingAverageFilter3D>(
                            mnncv::DEFAULT_GAUSSIAN_WINDOW_SIZE, mnncv::DEFAULT_GAUSSIAN_SIGMA);
                }
                break;
            case mnncv::HeadPoseFilterType::EMA:
                if (!s_head_pose_ema_filter) {
                    s_head_pose_ema_filter = std::make_unique<ExponentialMovingAverageFilter3D>(mnncv::DEFAULT_EMA_ALPHA);
                }
                break;
            case mnncv::HeadPoseFilterType::TIME_BASED_EMA:
                if (!s_head_pose_time_ema_filter) {
                    s_head_pose_time_ema_filter = std::make_unique<TimeBasedEMAFilter3D>(mnncv::DEFAULT_TIME_EMA_TAU_HEAD_POSE);
                }
                break;
        }
    }
}

[[maybe_unused]] mnncv::HeadPoseFilterType FaceLandmarker::getHeadPoseFilterType() {
    return s_head_pose_filter_type_;
}

void FaceLandmarker::setHeadPoseFilterTypeInt(int type) {
    if (type >= 0 && type <= 3) {
        setHeadPoseFilterType(static_cast<mnncv::HeadPoseFilterType>(type));
    }
}

void FaceLandmarker::setCameraFov(float fovDegrees) {
    if (fovDegrees > 0.0f && fovDegrees < 180.0f) {
        s_camera_fov_degrees_ = fovDegrees;
        if (s_geometry_initialized_ && s_geometry_pipeline_) {
            s_geometry_pipeline_.reset();
            s_geometry_initialized_ = false;
            initGeometryPipeline();
        }
    }
}

float FaceLandmarker::getCameraFov() {
    return s_camera_fov_degrees_;
}

bool FaceLandmarker::initGeometryPipeline() {
    if (s_geometry_initialized_) {
        return s_geometry_pipeline_ != nullptr;
    }
    
    s_geometry_initialized_ = true;
    
    if (s_head_pose_filter_enabled_) {
        setHeadPoseFilterType(s_head_pose_filter_type_);
    }
    
    auto metadata = MNN::FaceGeometry::CreateCanonicalFaceMeshMetadata();
    
    MNN::FaceGeometry::Environment env;
    env.origin_is_top_left = true;
    env.vertical_fov_degrees = s_camera_fov_degrees_;
    env.near = 1.0f;
    env.far = 10000.0f;
    
    s_geometry_pipeline_ = MNN::FaceGeometry::GeometryPipeline::Create(
        env, metadata.input_source, metadata.canonical_mesh, metadata.landmark_weights);
    
    if (!s_geometry_pipeline_) {
        FACE_LOGE("Failed to create Face Geometry pipeline");
        return false;
    }
    
    FACE_LOGI("Face Geometry pipeline initialized");
    return true;
}

[[maybe_unused]] MNN::FaceGeometry::GeometryPipeline* FaceLandmarker::getGeometryPipeline() {
    if (!s_geometry_initialized_) {
        initGeometryPipeline();
    }
    return s_geometry_pipeline_.get();
}

bool FaceLandmarker::estimateFaceGeometry(const std::vector<FaceLandmark>& landmarks,
                                           int image_width, int image_height,
                                           MNN::FaceGeometry::FaceGeometry& result) {
    if (landmarks.size() < 468) {
        return false;
    }
    
    if (!initGeometryPipeline() || !s_geometry_pipeline_) {
        return false;
    }
    
    std::vector<MNN::FaceGeometry::NormalizedLandmark> normalized_landmarks(468);
    for (int i = 0; i < 468; ++i) {
        normalized_landmarks[i].x = landmarks[i].x;
        normalized_landmarks[i].y = landmarks[i].y;
        normalized_landmarks[i].z = landmarks[i].z;
    }
    
    std::vector<std::vector<MNN::FaceGeometry::NormalizedLandmark>> faces = {normalized_landmarks};
    
    std::vector<MNN::FaceGeometry::FaceGeometry> results;
    bool success = s_geometry_pipeline_->EstimateFaceGeometry(
        faces, image_width, image_height, results);
    
    if (!success || results.empty()) {
        return false;
    }
    
    result = std::move(results[0]);
    return true;
}

HeadPose FaceLandmarker::estimateHeadPose(const std::vector<FaceLandmark>& landmarks,
                                           int image_width, int image_height) {
    HeadPose pose;
    pose.pitch = 0.0f;
    pose.yaw = 0.0f;
    pose.roll = 0.0f;
    pose.translation_x = 0.0f;
    pose.translation_y = 0.0f;
    pose.translation_z = 0.0f;
    pose.scale = 1.0f;
    
    if (landmarks.size() < 468) {
        return pose;
    }
    
    MNN::FaceGeometry::FaceGeometry geometry;
    if (!estimateFaceGeometry(landmarks, image_width, image_height, geometry)) {
        return pose;
    }
    
    extractEulerAnglesFromMatrix4f(geometry.pose_transform_matrix,
                                   pose.pitch, pose.yaw, pose.roll);
    
    if (s_head_pose_filter_enabled_) {
        switch (s_head_pose_filter_type_) {
            case mnncv::HeadPoseFilterType::ONE_EURO:
                if (s_head_pose_filter) {
                    s_head_pose_filter->filter(pose.pitch, pose.yaw, pose.roll);
                }
                break;
            case mnncv::HeadPoseFilterType::GAUSSIAN_MOVING_AVG:
                if (s_head_pose_gaussian_filter) {
                    s_head_pose_gaussian_filter->filter(pose.pitch, pose.yaw, pose.roll);
                }
                break;
            case mnncv::HeadPoseFilterType::EMA:
                if (s_head_pose_ema_filter) {
                    s_head_pose_ema_filter->filter(pose.pitch, pose.yaw, pose.roll);
                }
                break;
            case mnncv::HeadPoseFilterType::TIME_BASED_EMA:
                if (s_head_pose_time_ema_filter) {
                    s_head_pose_time_ema_filter->filter(pose.pitch, pose.yaw, pose.roll);
                }
                break;
        }
    }
    
    return pose;
}

} // namespace mediapipe_face
