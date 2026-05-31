/**
 * @file logging.h
 * @brief Android logging macros for MediaPipe Face pipeline
 * 
 * Provides unified logging interface across all modules.
 */

#ifndef MEDIAPIPE_FACE_LOGGING_H
#define MEDIAPIPE_FACE_LOGGING_H

#include <android/log.h>

// Default log tag - can be overridden before including this header
#ifndef FACE_LOG_TAG
#define FACE_LOG_TAG "MediaPipeFace"
#endif

// Log level macros
#define FACE_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, FACE_LOG_TAG, __VA_ARGS__)
#define FACE_LOGW(...) __android_log_print(ANDROID_LOG_WARN, FACE_LOG_TAG, __VA_ARGS__)
#define FACE_LOGI(...) __android_log_print(ANDROID_LOG_INFO, FACE_LOG_TAG, __VA_ARGS__)
#define FACE_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, FACE_LOG_TAG, __VA_ARGS__)
#define FACE_LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, FACE_LOG_TAG, __VA_ARGS__)

// Conditional logging (can be disabled for release builds)
#ifdef NDEBUG
#define FACE_LOGD_IF(cond, ...) ((void)0)
#define FACE_LOGV_IF(cond, ...) ((void)0)
#else
#define FACE_LOGD_IF(cond, ...) do { if (cond) FACE_LOGD(__VA_ARGS__); } while(0)
#define FACE_LOGV_IF(cond, ...) do { if (cond) FACE_LOGV(__VA_ARGS__); } while(0)
#endif

#endif // MEDIAPIPE_FACE_LOGGING_H
