/**
 * @file perf_profiler.h
 * @brief Performance profiling utilities for MediaPipe Face pipeline
 * 
 * Provides high-resolution timing and statistics collection for
 * profiling inference performance.
 */

#ifndef MEDIAPIPE_FACE_PERF_PROFILER_H
#define MEDIAPIPE_FACE_PERF_PROFILER_H

#include <chrono>
#include <algorithm>
#include "common/logging.h"

namespace mediapipe_face {

// ============================================================================
// Profiling Configuration
// ============================================================================

// Set to 1 to enable performance profiling (logs timing every N frames)
#ifndef ENABLE_PERF_PROFILING
#define ENABLE_PERF_PROFILING 1
#endif

#ifndef PERF_LOG_INTERVAL
#define PERF_LOG_INTERVAL 60  // Log stats every N frames
#endif

// ============================================================================
// Timer and Stats Classes
// ============================================================================

/**
 * @brief High-resolution timer for profiling
 */
class PerfTimer {
public:
    void start() { 
        start_ = std::chrono::high_resolution_clock::now(); 
    }
    
    [[nodiscard]] double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Accumulator for averaging timing statistics
 */
struct PerfStats {
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;
    int count = 0;
    
    void add(double ms) {
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        count++;
    }
    
    [[nodiscard]] double avg() const { 
        return count > 0 ? total_ms / count : 0.0; 
    }
    
    void reset() {
        total_ms = 0.0;
        min_ms = 1e9;
        max_ms = 0.0;
        count = 0;
    }
};

// ============================================================================
// Module-Specific Profilers
// ============================================================================

/**
 * @brief Face Detection performance profiler
 */
class DetectionPerfProfiler {
public:
    PerfStats total;
    PerfStats preprocess;
    PerfStats inference;
    PerfStats postprocess;
    PerfStats coord_transform;
    int frame_count = 0;
    
    void log() const {
        FACE_LOGI("=== Face Detection Perf (avg of %d frames) ===", frame_count);
        FACE_LOGI("  Total:      %.2f ms (min=%.2f, max=%.2f)", 
                  total.avg(), total.min_ms, total.max_ms);
        FACE_LOGI("  Preprocess: %.2f ms (%.1f%%)", 
                  preprocess.avg(), 100.0 * preprocess.avg() / total.avg());
        FACE_LOGI("  Inference:  %.2f ms (%.1f%%)", 
                  inference.avg(), 100.0 * inference.avg() / total.avg());
        FACE_LOGI("  Postproc:   %.2f ms (%.1f%%)", 
                  postprocess.avg(), 100.0 * postprocess.avg() / total.avg());
        FACE_LOGI("  CoordTrans: %.2f ms (%.1f%%)", 
                  coord_transform.avg(), 100.0 * coord_transform.avg() / total.avg());
    }
    
    void reset() {
        total.reset();
        preprocess.reset();
        inference.reset();
        postprocess.reset();
        coord_transform.reset();
        frame_count = 0;
    }
    
    void checkAndLog(int interval = PERF_LOG_INTERVAL) {
        if (++frame_count % interval == 0) {
            log();
            reset();
        }
    }
};

/**
 * @brief Face Landmark performance profiler
 */
class LandmarkPerfProfiler {
public:
    PerfStats total;
    PerfStats preprocess;
    PerfStats inference;
    PerfStats postprocess;
    PerfStats smoothing;
    int frame_count = 0;
    
    void log() const {
        FACE_LOGI("=== Face Landmarks Perf (avg of %d frames) ===", frame_count);
        FACE_LOGI("  Total:      %.2f ms (min=%.2f, max=%.2f)", 
                  total.avg(), total.min_ms, total.max_ms);
        FACE_LOGI("  Preprocess: %.2f ms (%.1f%%)", 
                  preprocess.avg(), 100.0 * preprocess.avg() / total.avg());
        FACE_LOGI("  Inference:  %.2f ms (%.1f%%)", 
                  inference.avg(), 100.0 * inference.avg() / total.avg());
        FACE_LOGI("  Postproc:   %.2f ms (%.1f%%)", 
                  postprocess.avg(), 100.0 * postprocess.avg() / total.avg());
        FACE_LOGI("  Smoothing:  %.2f ms (%.1f%%)", 
                  smoothing.avg(), 100.0 * smoothing.avg() / total.avg());
    }
    
    void reset() {
        total.reset();
        preprocess.reset();
        inference.reset();
        postprocess.reset();
        smoothing.reset();
        frame_count = 0;
    }
    
    void checkAndLog(int interval = PERF_LOG_INTERVAL) {
        if (++frame_count % interval == 0) {
            log();
            reset();
        }
    }
};

/**
 * @brief BlendShape prediction performance profiler
 */
class BlendShapePerfProfiler {
public:
    PerfStats total;
    PerfStats prepare_tensor;
    PerfStats inference;
    PerfStats compensation;
    int frame_count = 0;
    
    void log() const {
        FACE_LOGI("=== BlendShape Perf (avg of %d frames) ===", frame_count);
        FACE_LOGI("  Total:        %.2f ms (min=%.2f, max=%.2f)", 
                  total.avg(), total.min_ms, total.max_ms);
        FACE_LOGI("  PrepareTensor:%.2f ms (%.1f%%)", 
                  prepare_tensor.avg(), 100.0 * prepare_tensor.avg() / total.avg());
        FACE_LOGI("  Inference:    %.2f ms (%.1f%%)", 
                  inference.avg(), 100.0 * inference.avg() / total.avg());
        FACE_LOGI("  Compensation: %.2f ms (%.1f%%)", 
                  compensation.avg(), 100.0 * compensation.avg() / total.avg());
    }
    
    void reset() {
        total.reset();
        prepare_tensor.reset();
        inference.reset();
        compensation.reset();
        frame_count = 0;
    }
    
    void checkAndLog(int interval = PERF_LOG_INTERVAL) {
        if (++frame_count % interval == 0) {
            log();
            reset();
        }
    }
};

// ============================================================================
// Profiling Macros
// ============================================================================

#if ENABLE_PERF_PROFILING
    #define PERF_START(timer) mediapipe_face::PerfTimer timer; timer.start()
    #define PERF_END(timer, stats) stats.add(timer.elapsedMs())
    #define PERF_CHECK_LOG(profiler, interval) profiler.checkAndLog(interval)
#else
    #define PERF_START(timer) ((void)0)
    #define PERF_END(timer, stats) ((void)0)
    #define PERF_CHECK_LOG(profiler, interval) ((void)0)
#endif

} // namespace mediapipe_face

#endif // MEDIAPIPE_FACE_PERF_PROFILER_H
