/**
 * @file types.h
 * @brief Common data types for MediaPipe Face pipeline
 * 
 * Contains all shared data structures used across Face Detection,
 * Landmark Detection, and BlendShape Prediction modules.
 */

#ifndef MEDIAPIPE_FACE_TYPES_H
#define MEDIAPIPE_FACE_TYPES_H

#include <vector>
#include <cmath>
#include <cstring>

namespace mediapipe_face {

// ============================================================================
// Face Detection Types
// ============================================================================

/**
 * @brief Face detection result with bounding box and keypoints
 */
struct FaceDetection {
    float x1, y1, x2, y2;  ///< Bounding box (normalized to [0,1])
    float score;           ///< Detection confidence
    std::vector<std::pair<float, float>> keypoints;  ///< 6 keypoints (x, y) normalized
    int anchor_idx = -1;   ///< For debugging: which anchor generated this detection
};

/**
 * @brief SSD Anchor for face detection
 */
struct Anchor {
    float x_center;
    float y_center;
    float w;
    float h;
};

// ============================================================================
// Face Landmark Types
// ============================================================================

/**
 * @brief Single facial landmark point
 */
struct FaceLandmark {
    float x, y, z;  ///< Normalized coordinates [0,1] for x,y; z is relative depth
};

/**
 * @brief Face landmarks detection result
 */
struct FaceLandmarksResult {
    std::vector<FaceLandmark> landmarks;  ///< 478 landmarks (468 mesh + 10 iris)
    float presence_score;                  ///< Confidence that a face is present
    bool presence;                         ///< Whether face is present (based on threshold)

    bool isValid() const { return presence && !landmarks.empty(); }
};

// ============================================================================
// Head Pose Types
// ============================================================================

/**
 * @brief Head pose estimation result (Euler angles in degrees)
 */
struct HeadPose {
    float pitch;  ///< X-axis rotation: up(+) / down(-), range: [-90, 90]
    float yaw;    ///< Y-axis rotation: left(+) / right(-), range: [-90, 90]
    float roll;   ///< Z-axis rotation: tilt left(+) / right(-), range: [-180, 180]

    // Optional: translation and scale from pose matrix
    float translation_x = 0.0f;
    float translation_y = 0.0f;
    float translation_z = 0.0f;
    float scale = 1.0f;

    bool isValid() const {
        return !std::isnan(pitch) && !std::isnan(yaw) && !std::isnan(roll);
    }
};

// ============================================================================
// BlendShape Types
// ============================================================================

/**
 * @brief MediaPipe 52 BlendShape indices (ARKit compatible)
 */
enum BlendShapeIndex {
    BS_NEUTRAL = 0,
    BS_BROW_DOWN_LEFT = 1,
    BS_BROW_DOWN_RIGHT = 2,
    BS_BROW_INNER_UP = 3,
    BS_BROW_OUTER_UP_LEFT = 4,
    BS_BROW_OUTER_UP_RIGHT = 5,
    BS_CHEEK_PUFF = 6,
    BS_CHEEK_SQUINT_LEFT = 7,
    BS_CHEEK_SQUINT_RIGHT = 8,
    BS_EYE_BLINK_LEFT = 9,
    BS_EYE_BLINK_RIGHT = 10,
    BS_EYE_LOOK_DOWN_LEFT = 11,
    BS_EYE_LOOK_DOWN_RIGHT = 12,
    BS_EYE_LOOK_IN_LEFT = 13,
    BS_EYE_LOOK_IN_RIGHT = 14,
    BS_EYE_LOOK_OUT_LEFT = 15,
    BS_EYE_LOOK_OUT_RIGHT = 16,
    BS_EYE_LOOK_UP_LEFT = 17,
    BS_EYE_LOOK_UP_RIGHT = 18,
    BS_EYE_SQUINT_LEFT = 19,
    BS_EYE_SQUINT_RIGHT = 20,
    BS_EYE_WIDE_LEFT = 21,
    BS_EYE_WIDE_RIGHT = 22,
    BS_JAW_FORWARD = 23,
    BS_JAW_LEFT = 24,
    BS_JAW_OPEN = 25,
    BS_JAW_RIGHT = 26,
    BS_MOUTH_CLOSE = 27,
    BS_MOUTH_DIMPLE_LEFT = 28,
    BS_MOUTH_DIMPLE_RIGHT = 29,
    BS_MOUTH_FROWN_LEFT = 30,
    BS_MOUTH_FROWN_RIGHT = 31,
    BS_MOUTH_FUNNEL = 32,
    BS_MOUTH_LEFT = 33,
    BS_MOUTH_LOWER_DOWN_LEFT = 34,
    BS_MOUTH_LOWER_DOWN_RIGHT = 35,
    BS_MOUTH_PRESS_LEFT = 36,
    BS_MOUTH_PRESS_RIGHT = 37,
    BS_MOUTH_PUCKER = 38,
    BS_MOUTH_RIGHT = 39,
    BS_MOUTH_ROLL_LOWER = 40,
    BS_MOUTH_ROLL_UPPER = 41,
    BS_MOUTH_SHRUG_LOWER = 42,
    BS_MOUTH_SHRUG_UPPER = 43,
    BS_MOUTH_SMILE_LEFT = 44,
    BS_MOUTH_SMILE_RIGHT = 45,
    BS_MOUTH_STRETCH_LEFT = 46,
    BS_MOUTH_STRETCH_RIGHT = 47,
    BS_MOUTH_UPPER_UP_LEFT = 48,
    BS_MOUTH_UPPER_UP_RIGHT = 49,
    BS_NOSE_SNEER_LEFT = 50,
    BS_NOSE_SNEER_RIGHT = 51,
    BS_COUNT = 52
};

/**
 * @brief BlendShape names for Live2D mapping
 */
inline const char* getBlendShapeName(BlendShapeIndex idx) {
    static const char* kBlendShapeNames[BS_COUNT] = {
        "_neutral",
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "noseSneerLeft",
        "noseSneerRight"
    };
    return (idx >= 0 && idx < BS_COUNT) ? kBlendShapeNames[idx] : "unknown";
}

/**
 * @brief BlendShape prediction result
 */
struct BlendShapeResult {
    float values[BS_COUNT];       ///< Raw blendshape values [0, 1]
    float compensated[BS_COUNT];  ///< Values after head rotation compensation
    bool valid;
    
    BlendShapeResult() : valid(false) {
        std::memset(values, 0, sizeof(values));
        std::memset(compensated, 0, sizeof(compensated));
    }
    
    float get(BlendShapeIndex idx) const { return compensated[idx]; }
    float getRaw(BlendShapeIndex idx) const { return values[idx]; }
    const char* getName(BlendShapeIndex idx) const { return getBlendShapeName(idx); }
};

} // namespace mediapipe_face

#endif // MEDIAPIPE_FACE_TYPES_H
