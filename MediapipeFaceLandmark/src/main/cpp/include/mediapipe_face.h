/**
 * @file mediapipe_face.h
 * @brief Unified header for MediaPipe Face pipeline
 * 
 * This is the main include file for using the MediaPipe Face pipeline.
 * It includes all necessary headers for face detection, landmark detection,
 * and blendshape prediction.
 * 
 * Usage:
 *   #include "mediapipe_face.h"
 *   using namespace mediapipe_face;
 *   
 *   FaceDetector detector;
 *   FaceLandmarker landmarker;
 *   BlendShapePredictor blendshape;
 */

#ifndef MEDIAPIPE_FACE_H
#define MEDIAPIPE_FACE_H

// Common utilities
#include "common/types.h"
#include "common/logging.h"
#include "common/perf_profiler.h"

// Image processing utilities
#include "utils/yuv_utils.h"

// Core modules
#include "face_detector.h"
#include "face_landmarker.h"
#include "blendshape_predictor.h"

#endif // MEDIAPIPE_FACE_H
