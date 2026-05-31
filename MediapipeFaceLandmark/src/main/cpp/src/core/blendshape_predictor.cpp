/**
 * @file blendshape_predictor.cpp
 * @brief BlendShape Prediction implementation using MediaPipe model
 * 
 * Migrated from simple_mediapipe_face.cpp (SimpleBlendShapePredictor class)
 */

#include "blendshape_predictor.h"
#include "common/logging.h"
#include "common/perf_profiler.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#undef FACE_LOG_TAG
#define FACE_LOG_TAG "BlendShapePredictor"

namespace mediapipe_face {

// Performance profiler
static BlendShapePerfProfiler s_perf_profiler;

// ============================================================================
// Landmark Subset Indices
// ============================================================================

// Subset of 478 landmarks required for BlendShape model (146 points)
// From face_blendshapes_graph.cc kLandmarksSubsetIdxs
const int BlendShapePredictor::kLandmarkSubsetIndices[146] = {
    0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,
    40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
    81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
};

// ============================================================================
// BlendShapePredictor Implementation
// ============================================================================

BlendShapePredictor::BlendShapePredictor() = default;

BlendShapePredictor::~BlendShapePredictor() {
    if (interpreter_) {
        interpreter_->releaseModel();
        if (session_) {
            interpreter_->releaseSession(session_);
        }
    }
}

bool BlendShapePredictor::loadModel(const std::string& mnn_path, int num_threads) {
    last_error_.clear();
    
    interpreter_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(mnn_path.c_str()));
    if (!interpreter_) {
        last_error_ = "createFromFile failed for BlendShape model";
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
        last_error_ = "createSession failed for BlendShape model";
        return false;
    }
    
    input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor_) {
        last_error_ = "getSessionInput failed for BlendShape model";
        return false;
    }
    
    auto shape = input_tensor_->shape();
    if (shape.size() >= 2) {
        input_h_ = shape[1];  // Should be 146
        input_w_ = shape.size() > 2 ? shape[2] : 2;  // Should be 2
    }
    
    FACE_LOGI("BlendShape model loaded: input [%d, %d]", input_h_, input_w_);
    return true;
}

void BlendShapePredictor::prepareLandmarkTensor(
    const std::vector<FaceLandmark>& landmarks,
    int image_width, int image_height,
    float* output) {
    
    // MediaPipe LandmarksToTensorCalculator:
    // 1. Extracts X, Y attributes from landmarks
    // 2. Denormalizes by image size (pixel coordinates)
    // 3. Output shape: [1, 146, 2]
    
    for (int i = 0; i < kLandmarkSubsetSize; ++i) {
        int lm_idx = kLandmarkSubsetIndices[i];
        if (lm_idx < static_cast<int>(landmarks.size())) {
            output[i * 2 + 0] = landmarks[lm_idx].x * static_cast<float>(image_width);
            output[i * 2 + 1] = landmarks[lm_idx].y * static_cast<float>(image_height);
        } else {
            output[i * 2 + 0] = 0.0f;
            output[i * 2 + 1] = 0.0f;
        }
    }
}

BlendShapeResult BlendShapePredictor::predict(
    const std::vector<FaceLandmark>& landmarks,
    int image_width, int image_height,
    const HeadPose* head_pose) {
    
    PERF_START(total_timer);
    BlendShapeResult result;
    
    if (!interpreter_ || !session_ || !input_tensor_) {
        last_error_ = "BlendShape model not loaded";
        return result;
    }
    
    if (landmarks.size() < 468) {
        last_error_ = "Insufficient landmarks for BlendShape prediction";
        return result;
    }
    
    // Prepare Tensor
    PERF_START(prepare_timer);
    
    std::vector<float> input_data(kLandmarkSubsetSize * 2);
    prepareLandmarkTensor(landmarks, image_width, image_height, input_data.data());
    
    auto input_host = new MNN::Tensor(input_tensor_, input_tensor_->getDimensionType());
    std::memcpy(input_host->host<float>(), input_data.data(),
                input_data.size() * sizeof(float));
    input_tensor_->copyFromHostTensor(input_host);
    delete input_host;
    
    PERF_END(prepare_timer, s_perf_profiler.prepare_tensor);
    
    // Inference
    PERF_START(inference_timer);
    interpreter_->runSession(session_);
    PERF_END(inference_timer, s_perf_profiler.inference);
    
    // Get output
    auto output_tensor = interpreter_->getSessionOutput(session_, nullptr);
    if (!output_tensor) {
        auto outputs = interpreter_->getSessionOutputAll(session_);
        if (!outputs.empty()) {
            output_tensor = outputs.begin()->second;
        }
    }
    
    if (!output_tensor) {
        last_error_ = "Failed to get BlendShape output tensor";
        return result;
    }
    
    std::shared_ptr<MNN::Tensor> output_host(
        new MNN::Tensor(output_tensor, MNN::Tensor::CAFFE));
    output_tensor->copyToHostTensor(output_host.get());
    
    const float* output_data = output_host->host<float>();
    int output_size = output_host->elementSize();
    
    int num_blendshapes = std::min(output_size, static_cast<int>(BS_COUNT));
    for (int i = 0; i < num_blendshapes; ++i) {
        result.values[i] = std::max(0.0f, std::min(1.0f, output_data[i]));
        result.compensated[i] = result.values[i];
    }
    
    result.valid = true;
    
    // Compensation
    PERF_START(compensation_timer);
    
    if (head_pose && head_pose->isValid()) {
        applyHeadRotationCompensation(result, *head_pose);
    }
    
    PERF_END(compensation_timer, s_perf_profiler.compensation);
    PERF_END(total_timer, s_perf_profiler.total);
    PERF_CHECK_LOG(s_perf_profiler, PERF_LOG_INTERVAL);
    
    return result;
}

void BlendShapePredictor::applyHeadRotationCompensation(
    BlendShapeResult& result,
    const HeadPose& pose) {
    
    // Head rotation compensation based on OpenSeeFace approach
    const float pitch = pose.pitch;
    const float yaw = pose.yaw;
    
    // 1. Eye blink compensation for looking down
    if (pitch > 10.0f) {
        float compensation = std::min(0.3f, (pitch - 10.0f) / 30.0f);
        result.compensated[BS_EYE_BLINK_LEFT] = 
            std::max(0.0f, result.values[BS_EYE_BLINK_LEFT] - compensation);
        result.compensated[BS_EYE_BLINK_RIGHT] = 
            std::max(0.0f, result.values[BS_EYE_BLINK_RIGHT] - compensation);
    }
    
    // 2. Eyebrow compensation for pitch
    if (std::abs(pitch) > 10.0f) {
        float brow_compensation = std::max(-0.2f, std::min(0.2f, -pitch / 60.0f));
        result.compensated[BS_BROW_INNER_UP] = 
            std::max(0.0f, std::min(1.0f, result.values[BS_BROW_INNER_UP] + brow_compensation));
        result.compensated[BS_BROW_OUTER_UP_LEFT] = 
            std::max(0.0f, std::min(1.0f, result.values[BS_BROW_OUTER_UP_LEFT] + brow_compensation));
        result.compensated[BS_BROW_OUTER_UP_RIGHT] = 
            std::max(0.0f, std::min(1.0f, result.values[BS_BROW_OUTER_UP_RIGHT] + brow_compensation));
    }
    
    // 3. Mouth corner compensation for yaw
    if (std::abs(yaw) > 15.0f) {
        float yaw_factor = std::min(0.3f, std::abs(yaw) / 45.0f);
        
        if (yaw > 0) {
            result.compensated[BS_MOUTH_SMILE_RIGHT] = 
                std::max(0.0f, result.values[BS_MOUTH_SMILE_RIGHT] - yaw_factor * 0.3f);
            result.compensated[BS_MOUTH_FROWN_RIGHT] = 
                std::max(0.0f, result.values[BS_MOUTH_FROWN_RIGHT] - yaw_factor * 0.3f);
        } else {
            result.compensated[BS_MOUTH_SMILE_LEFT] = 
                std::max(0.0f, result.values[BS_MOUTH_SMILE_LEFT] - yaw_factor * 0.3f);
            result.compensated[BS_MOUTH_FROWN_LEFT] = 
                std::max(0.0f, result.values[BS_MOUTH_FROWN_LEFT] - yaw_factor * 0.3f);
        }
    }
    
    // 4. Cheek squint compensation
    if (std::abs(yaw) > 20.0f) {
        float squint_compensation = std::min(0.2f, (std::abs(yaw) - 20.0f) / 40.0f);
        
        if (yaw > 0) {
            result.compensated[BS_CHEEK_SQUINT_RIGHT] = 
                std::max(0.0f, result.values[BS_CHEEK_SQUINT_RIGHT] - squint_compensation);
        } else {
            result.compensated[BS_CHEEK_SQUINT_LEFT] = 
                std::max(0.0f, result.values[BS_CHEEK_SQUINT_LEFT] - squint_compensation);
        }
    }
    
    // 5. Eye look direction compensation
    if (yaw > 10.0f) {
        float reduce = std::min(0.5f, (yaw - 10.0f) / 30.0f);
        result.compensated[BS_EYE_LOOK_OUT_LEFT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_OUT_LEFT] - reduce);
        result.compensated[BS_EYE_LOOK_IN_RIGHT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_IN_RIGHT] - reduce);
    } else if (yaw < -10.0f) {
        float reduce = std::min(0.5f, (-yaw - 10.0f) / 30.0f);
        result.compensated[BS_EYE_LOOK_OUT_RIGHT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_OUT_RIGHT] - reduce);
        result.compensated[BS_EYE_LOOK_IN_LEFT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_IN_LEFT] - reduce);
    }
    
    if (pitch > 10.0f) {
        float reduce = std::min(0.5f, (pitch - 10.0f) / 30.0f);
        result.compensated[BS_EYE_LOOK_DOWN_LEFT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_DOWN_LEFT] - reduce);
        result.compensated[BS_EYE_LOOK_DOWN_RIGHT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_DOWN_RIGHT] - reduce);
    } else if (pitch < -10.0f) {
        float reduce = std::min(0.5f, (-pitch - 10.0f) / 30.0f);
        result.compensated[BS_EYE_LOOK_UP_LEFT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_UP_LEFT] - reduce);
        result.compensated[BS_EYE_LOOK_UP_RIGHT] = 
            std::max(0.0f, result.values[BS_EYE_LOOK_UP_RIGHT] - reduce);
    }
}

} // namespace mediapipe_face
