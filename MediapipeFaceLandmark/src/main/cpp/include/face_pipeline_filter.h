//
// One Euro Filter Implementation
// Reference: https://cristal.univ-lille.fr/~casiez/1euro/
//
// This filter is used by MediaPipe to smooth landmark positions and reduce jitter.
// It adapts the smoothing based on the velocity of the signal:
// - Low velocity = more smoothing (reduces noise when stationary)
// - High velocity = less smoothing (preserves fast movements)
//

#ifndef FACE_PIPELINE_FILTER_H
#define FACE_PIPELINE_FILTER_H

#include <cmath>
#include <vector>
#include <chrono>
#include <set>

namespace mnncv {

/**
 * Low-pass filter with adjustable alpha
 */
class LowPassFilter {
public:
    explicit LowPassFilter(float alpha = 1.0f) : alpha_(alpha), initialized_(false), value_(0.0f) {}
    
    float filter(float value, float alpha) {
        if (!initialized_) {
            initialized_ = true;
            value_ = value;
            return value;
        }
        value_ = alpha * value + (1.0f - alpha) * value_;
        return value_;
    }
    
    float lastValue() const { return value_; }
    bool hasLastValue() const { return initialized_; }
    void reset() { initialized_ = false; }
    
private:
    float alpha_;
    bool initialized_;
    float value_;
};

/**
 * One Euro Filter for scalar values
 * 
 * Parameters (from MediaPipe face_landmarks_detector_graph.cc):
 * - min_cutoff: 0.05 (minimum cutoff frequency, lower = more smoothing)
 * - beta: 80.0 (speed coefficient, higher = less lag during fast movements)
 * - derivate_cutoff: 1.0 (cutoff frequency for derivative filter)
 */
class OneEuroFilter {
public:
    explicit OneEuroFilter(float min_cutoff = 0.05f,
                  float beta = 80.0f, 
                  float derivate_cutoff = 1.0f)
        : min_cutoff_(min_cutoff)
        , beta_(beta)
        , derivate_cutoff_(derivate_cutoff)
        , last_time_(-1.0)
        , dx_filter_(computeAlpha(derivate_cutoff_, 1.0f / 30.0f))
        , x_filter_(1.0f) {}
    
    /**
     * Filter a value with automatic time delta calculation
     * @param value Current value to filter
     * @param timestamp Current timestamp in seconds (or use -1 for auto)
     * @return Filtered value
     */
    float filter(float value, double timestamp = -1.0) {
        // Use current time if timestamp not provided
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }
        
        float dt = (last_time_ < 0) ? (1.0f / 30.0f) : static_cast<float>(timestamp - last_time_);
        last_time_ = timestamp;
        
        // Clamp dt to reasonable range
        dt = std::max(0.001f, std::min(1.0f, dt));
        
        // Compute derivative
        float dx = x_filter_.hasLastValue() ? (value - x_filter_.lastValue()) / dt : 0.0f;
        
        // Filter derivative
        float dx_alpha = computeAlpha(derivate_cutoff_, dt);
        float edx = dx_filter_.filter(dx, dx_alpha);
        
        // Compute cutoff frequency based on velocity
        float cutoff = min_cutoff_ + beta_ * std::abs(edx);
        
        // Filter value
        float alpha = computeAlpha(cutoff, dt);
        return x_filter_.filter(value, alpha);
    }
    
    void reset() {
        last_time_ = -1.0;
        dx_filter_.reset();
        x_filter_.reset();
    }
    
private:
    static float computeAlpha(float cutoff, float dt) {
        float tau = 1.0f / (2.0f * (float)M_PI * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }
    
    float min_cutoff_;
    float beta_;
    float derivate_cutoff_;
    double last_time_;
    LowPassFilter dx_filter_;
    LowPassFilter x_filter_;
};

/**
 * One Euro Filter for 2D points (x, y)
 */
class OneEuroFilter2D {
public:
    explicit OneEuroFilter2D(float min_cutoff = 0.05f,
                    float beta = 80.0f, 
                    float derivate_cutoff = 1.0f)
        : x_filter_(min_cutoff, beta, derivate_cutoff)
        , y_filter_(min_cutoff, beta, derivate_cutoff) {}
    
    void filter(float& x, float& y, double timestamp = -1.0) {
        x = x_filter_.filter(x, timestamp);
        y = y_filter_.filter(y, timestamp);
    }
    
    void reset() {
        x_filter_.reset();
        y_filter_.reset();
    }
    
private:
    OneEuroFilter x_filter_;
    OneEuroFilter y_filter_;
};

/**
 * One Euro Filter for 3D points (x, y, z)
 */
class OneEuroFilter3D {
public:
    explicit OneEuroFilter3D(float min_cutoff = 0.05f,
                    float beta = 80.0f, 
                    float derivate_cutoff = 1.0f)
        : x_filter_(min_cutoff, beta, derivate_cutoff)
        , y_filter_(min_cutoff, beta, derivate_cutoff)
        , z_filter_(min_cutoff, beta, derivate_cutoff) {}
    
    void filter(float& x, float& y, float& z, double timestamp = -1.0) {
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }
        x = x_filter_.filter(x, timestamp);
        y = y_filter_.filter(y, timestamp);
        z = z_filter_.filter(z, timestamp);
    }
    
    void reset() {
        x_filter_.reset();
        y_filter_.reset();
        z_filter_.reset();
    }
    
private:
    OneEuroFilter x_filter_;
    OneEuroFilter y_filter_;
    OneEuroFilter z_filter_;
};

/**
 * Landmarks Smoothing Filter
 * Applies One Euro Filter to all landmarks (478 points for face mesh)
 * 
 * MediaPipe parameters:
 * - min_cutoff: 0.05
 * - beta: 80.0
 * - derivate_cutoff: 1.0
 */
class LandmarksSmoothingFilter {
public:
    explicit LandmarksSmoothingFilter(int num_landmarks = 478,
                             float min_cutoff = 0.05f,
                             float beta = 80.0f,
                             float derivate_cutoff = 1.0f)
        : num_landmarks_(num_landmarks)
        , min_cutoff_(min_cutoff)
        , beta_(beta)
        , derivate_cutoff_(derivate_cutoff) {
        // Initialize filters for each landmark
        filters_.reserve(num_landmarks);
        for (int i = 0; i < num_landmarks; ++i) {
            filters_.emplace_back(min_cutoff, beta, derivate_cutoff);
        }
    }

    /**
     * Apply smoothing to landmarks in-place
     * @param landmarks Vector of landmarks (will be modified)
     * @param timestamp Current timestamp in seconds (optional)
     */
    template<typename LandmarkType>
    void apply(std::vector<LandmarkType>& landmarks, double timestamp = -1.0) {
        // Use current time if not provided
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }
        
        int count = std::min(static_cast<int>(landmarks.size()), num_landmarks_);
        for (int i = 0; i < count; ++i) {
            filters_[i].filter(landmarks[i].x, landmarks[i].y, landmarks[i].z, timestamp);
        }
    }
    
    void reset() {
        for (auto& filter : filters_) {
            filter.reset();
        }
    }
    
    // Update filter parameters
    void setParameters(float min_cutoff, float beta, float derivate_cutoff) {
        min_cutoff_ = min_cutoff;
        beta_ = beta;
        derivate_cutoff_ = derivate_cutoff;
        // Recreate filters with new parameters
        filters_.clear();
        filters_.reserve(num_landmarks_);
        for (int i = 0; i < num_landmarks_; ++i) {
            filters_.emplace_back(min_cutoff, beta, derivate_cutoff);
        }
    }
    
private:
    int num_landmarks_;
    float min_cutoff_;
    float beta_;
    float derivate_cutoff_;
    std::vector<OneEuroFilter3D> filters_;
};

/**
 * Detection Box Smoothing Filter
 * Applies One Euro Filter to face detection boxes
 */
class DetectionSmoothingFilter {
public:
    explicit DetectionSmoothingFilter(float min_cutoff = 0.1f,  // Slightly higher for boxes
                             float beta = 40.0f,
                             float derivate_cutoff = 1.0f)
        : x1_filter_(min_cutoff, beta, derivate_cutoff)
        , y1_filter_(min_cutoff, beta, derivate_cutoff)
        , x2_filter_(min_cutoff, beta, derivate_cutoff)
        , y2_filter_(min_cutoff, beta, derivate_cutoff) {}
    
    void apply(float& x1, float& y1, float& x2, float& y2, double timestamp = -1.0) {
        x1 = x1_filter_.filter(x1, timestamp);
        y1 = y1_filter_.filter(y1, timestamp);
        x2 = x2_filter_.filter(x2, timestamp);
        y2 = y2_filter_.filter(y2, timestamp);
    }
    
    void reset() {
        x1_filter_.reset();
        y1_filter_.reset();
        x2_filter_.reset();
        y2_filter_.reset();
    }
    
private:
    OneEuroFilter x1_filter_;
    OneEuroFilter y1_filter_;
    OneEuroFilter x2_filter_;
    OneEuroFilter y2_filter_;
};

/**
 * Gaussian Weighted Moving Average Filter for 3D values
 *
 * Unlike One Euro Filter which adapts based on velocity, this filter
 * uses a fixed-size window with Gaussian weights to smooth values.
 * Better suited for signals that have already been pre-filtered (like head pose
 * derived from smoothed landmarks).
 *
 * The Gaussian weights give more importance to recent samples while still
 * considering history, providing stable smoothing without the velocity-based
 * adaptation that can fail on pre-filtered signals.
 */
class GaussianMovingAverageFilter {
public:
    /**
     * @param window_size Number of samples to keep in history (default 5)
     * @param sigma Gaussian sigma for weight calculation (default 1.5)
     */
    explicit GaussianMovingAverageFilter(int window_size = 5, float sigma = 1.5f)
        : window_size_(window_size)
        , sigma_(sigma) {
        // Pre-compute Gaussian weights
        weights_.resize(window_size);
        float sum = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            // Distance from most recent sample (index 0 = most recent)
            auto dist = static_cast<float>(i);
            weights_[i] = std::exp(-(dist * dist) / (2.0f * sigma * sigma));
            sum += weights_[i];
        }
        // Normalize weights
        for (int i = 0; i < window_size; ++i) {
            weights_[i] /= sum;
        }
    }

    float filter(float value) {
        // Add new value to front
        history_.insert(history_.begin(), value);

        // Keep only window_size samples
        if (static_cast<int>(history_.size()) > window_size_) {
            history_.pop_back();
        }

        // Compute weighted average
        float result = 0.0f;
        float weight_sum = 0.0f;
        for (size_t i = 0; i < history_.size(); ++i) {
            result += weights_[i] * history_[i];
            weight_sum += weights_[i];
        }

        return result / weight_sum;
    }

    void reset() {
        history_.clear();
    }

private:
    int window_size_;
    float sigma_;
    std::vector<float> weights_;
    std::vector<float> history_;
};

/**
 * Gaussian Moving Average Filter for 3D points (pitch, yaw, roll)
 */
class GaussianMovingAverageFilter3D {
public:
    explicit GaussianMovingAverageFilter3D(int window_size = 5, float sigma = 1.5f)
        : x_filter_(window_size, sigma)
        , y_filter_(window_size, sigma)
        , z_filter_(window_size, sigma) {}

    void filter(float& x, float& y, float& z) {
        x = x_filter_.filter(x);
        y = y_filter_.filter(y);
        z = z_filter_.filter(z);
    }

    void reset() {
        x_filter_.reset();
        y_filter_.reset();
        z_filter_.reset();
    }

private:
    GaussianMovingAverageFilter x_filter_;
    GaussianMovingAverageFilter y_filter_;
    GaussianMovingAverageFilter z_filter_;
};

/**
 * Exponential Moving Average Filter (EMA) for scalar values
 *
 * Simpler alternative to One Euro Filter. Uses fixed alpha regardless of velocity.
 * Good for post-processing already-smoothed signals.
 *
 * Formula: output = alpha * input + (1 - alpha) * previous_output
 */
class ExponentialMovingAverageFilter {
public:
    /**
     * @param alpha Smoothing factor (0 < alpha <= 1)
     *              Lower alpha = more smoothing but more lag
     *              Typical values: 0.1 (heavy), 0.3 (medium), 0.5 (light)
     */
    explicit ExponentialMovingAverageFilter(float alpha = 0.2f)
        : alpha_(alpha)
        , initialized_(false)
        , value_(0.0f) {}

    float filter(float value) {
        if (!initialized_) {
            initialized_ = true;
            value_ = value;
            return value;
        }
        value_ = alpha_ * value + (1.0f - alpha_) * value_;
        return value_;
    }

    void reset() {
        initialized_ = false;
    }

    void setAlpha(float alpha) {
        alpha_ = std::max(0.01f, std::min(1.0f, alpha));
    }

private:
    float alpha_;
    bool initialized_;
    float value_;
};

/**
 * Time-based Exponential Moving Average Filter
 *
 * Unlike fixed-alpha EMA, this version uses a time constant (tau) to determine
 * smoothing strength, making it frame-rate independent.
 *
 * Formula: alpha = 1 - exp(-dt / tau)
 *
 * With this approach:
 * - tau = time constant in seconds (time to reach ~63% of target)
 * - Higher tau = more smoothing
 * - Smoothing behavior is consistent regardless of frame rate
 */
class TimeBasedEMAFilter {
public:
    /**
     * @param tau Time constant in seconds
     *            0.05 = very responsive (reaches 63% in 50ms)
     *            0.10 = responsive
     *            0.15 = moderate smoothing
     *            0.20 = heavy smoothing (reaches 63% in 200ms)
     */
    explicit TimeBasedEMAFilter(float tau = 0.1f)
        : tau_(tau)
        , initialized_(false)
        , value_(0.0f)
        , last_time_(-1.0) {}

    float filter(float value, double timestamp = -1.0) {
        // Use current time if not provided
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }

        if (!initialized_) {
            initialized_ = true;
            value_ = value;
            last_time_ = timestamp;
            return value;
        }

        // Calculate time delta
        auto dt = static_cast<float>(timestamp - last_time_);
        last_time_ = timestamp;

        // Clamp dt to avoid numerical issues
        dt = std::max(0.001f, std::min(0.5f, dt));

        // Calculate frame-rate independent alpha
        // alpha = 1 - exp(-dt / tau)
        float alpha = 1.0f - std::exp(-dt / tau_);

        value_ = alpha * value + (1.0f - alpha) * value_;
        return value_;
    }

    void reset() {
        initialized_ = false;
        last_time_ = -1.0;
    }

    void setTau(float tau) {
        tau_ = std::max(0.01f, tau);
    }

    [[nodiscard]] float getTau() const { return tau_; }

private:
    float tau_;
    bool initialized_;
    float value_;
    double last_time_;
};

/**
 * Time-based EMA Filter for 3D points
 */
class TimeBasedEMAFilter3D {
public:
    explicit TimeBasedEMAFilter3D(float tau = 0.1f)
        : x_filter_(tau)
        , y_filter_(tau)
        , z_filter_(tau) {}

    void filter(float& x, float& y, float& z, double timestamp = -1.0) {
        // Share timestamp across all 3 dimensions
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }
        x = x_filter_.filter(x, timestamp);
        y = y_filter_.filter(y, timestamp);
        z = z_filter_.filter(z, timestamp);
    }

    void reset() {
        x_filter_.reset();
        y_filter_.reset();
        z_filter_.reset();
    }

    void setTau(float tau) {
        x_filter_.setTau(tau);
        y_filter_.setTau(tau);
        z_filter_.setTau(tau);
    }

private:
    TimeBasedEMAFilter x_filter_;
    TimeBasedEMAFilter y_filter_;
    TimeBasedEMAFilter z_filter_;
};

/**
 * Exponential Moving Average Filter for 3D points
 */
class ExponentialMovingAverageFilter3D {
public:
    explicit ExponentialMovingAverageFilter3D(float alpha = 0.2f)
        : x_filter_(alpha)
        , y_filter_(alpha)
        , z_filter_(alpha) {}

    void filter(float& x, float& y, float& z) {
        x = x_filter_.filter(x);
        y = y_filter_.filter(y);
        z = z_filter_.filter(z);
    }

    void reset() {
        x_filter_.reset();
        y_filter_.reset();
        z_filter_.reset();
    }

    void setAlpha(float alpha) {
        x_filter_.setAlpha(alpha);
        y_filter_.setAlpha(alpha);
        z_filter_.setAlpha(alpha);
    }

private:
    ExponentialMovingAverageFilter x_filter_;
    ExponentialMovingAverageFilter y_filter_;
    ExponentialMovingAverageFilter z_filter_;
};

// ============================================================================
// Filter Type Enumeration (for JNI control)
// ============================================================================

/**
 * detection && landmark Filter type enumeration - can be set from Java layer via JNI
 */
enum class SmoothingFilterType {
    ONE_EURO = 0,        // Original MediaPipe filter (velocity adaptive)
    TIME_BASED_EMA = 1,  // Frame-rate independent EMA (recommended for high FPS)
    NONE = -1             // No smoothing
};

// Head pose filter type
enum class HeadPoseFilterType {
    ONE_EURO,           // Adaptive filter based on velocity (original)
    GAUSSIAN_MOVING_AVG, // Fixed window with Gaussian weights (good for pre-filtered signals)
    EMA,                // Simple exponential moving average (simplest, most stable)
    TIME_BASED_EMA      // Frame-rate independent EMA (recommended)
};

/**
 * Filter configuration parameters
 */
static constexpr float DEFAULT_TIME_EMA_TAU = 0.12f; // (time_ema_tau * 1000) ms time constant
static constexpr float DEFAULT_TIME_EMA_TAU_HEAD_POSE = 0.3f;
static constexpr int DEFAULT_GAUSSIAN_WINDOW_SIZE = 5;
static constexpr float DEFAULT_GAUSSIAN_SIGMA = 1.5f;
static constexpr float DEFAULT_EMA_ALPHA = 0.3f;
static constexpr float DEFAULT_ONE_EURO_MIN_CUTOFF = 0.05f;
static constexpr float DEFAULT_ONE_EURO_BETA = 80.0f;
static constexpr float DEFAULT_ONE_EURO_DERIVATE_CUTOFF = 1.0f;
struct FilterConfig {
    SmoothingFilterType type = SmoothingFilterType::TIME_BASED_EMA;

    // One Euro Filter params
    float one_euro_min_cutoff = DEFAULT_ONE_EURO_MIN_CUTOFF;
    float one_euro_beta = DEFAULT_ONE_EURO_BETA;
    float one_euro_derivate_cutoff = DEFAULT_ONE_EURO_DERIVATE_CUTOFF;

    // Time-Based EMA params
    float time_ema_tau = DEFAULT_TIME_EMA_TAU;

    // Gaussian params
    int gaussian_window_size = DEFAULT_GAUSSIAN_WINDOW_SIZE;
    float gaussian_sigma = DEFAULT_GAUSSIAN_SIGMA;

    // Simple EMA params
    float ema_alpha = DEFAULT_EMA_ALPHA;
};


// ============================================================================
// Time-Based EMA Landmarks Smoothing Filter
// ============================================================================

/**
 * Landmarks Smoothing Filter using Time-Based EMA
 * Frame-rate independent version that works well at any FPS
 */
class TimeBasedLandmarksSmoothingFilter {
public:
    explicit TimeBasedLandmarksSmoothingFilter(int num_landmarks = 478, float tau = 0.08f)
        : num_landmarks_(num_landmarks), tau_(tau) {
        filters_.reserve(num_landmarks);
        for (int i = 0; i < num_landmarks; ++i) {
            filters_.emplace_back(tau);
        }
    }

    template<typename LandmarkType>
    void apply(std::vector<LandmarkType>& landmarks, double timestamp = -1.0) {
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }

        int count = std::min(static_cast<int>(landmarks.size()), num_landmarks_);
        for (int i = 0; i < count; ++i) {
            filters_[i].filter(landmarks[i].x, landmarks[i].y, landmarks[i].z, timestamp);
        }
    }

    void reset() {
        for (auto& filter : filters_) {
            filter.reset();
        }
    }

    void setTau(float tau) {
        tau_ = tau;
        for (auto& filter : filters_) {
            filter.setTau(tau);
        }
    }

    [[nodiscard]] float getTau() const { return tau_; }

private:
    int num_landmarks_;
    float tau_;
    std::vector<TimeBasedEMAFilter3D> filters_;
};

// ============================================================================
// Time-Based EMA Detection Smoothing Filter
// ============================================================================

/**
 * Detection Box Smoothing Filter using Time-Based EMA
 * Frame-rate independent version
 */
class TimeBasedDetectionSmoothingFilter {
public:
    explicit TimeBasedDetectionSmoothingFilter(float tau = 0.06f)  // 60ms for faster response
        : x1_filter_(tau)
        , y1_filter_(tau)
        , x2_filter_(tau)
        , y2_filter_(tau) {}

    void apply(float& x1, float& y1, float& x2, float& y2, double timestamp = -1.0) {
        if (timestamp < 0) {
            auto now = std::chrono::steady_clock::now();
            timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        }
        x1 = x1_filter_.filter(x1, timestamp);
        y1 = y1_filter_.filter(y1, timestamp);
        x2 = x2_filter_.filter(x2, timestamp);
        y2 = y2_filter_.filter(y2, timestamp);
    }

    void reset() {
        x1_filter_.reset();
        y1_filter_.reset();
        x2_filter_.reset();
        y2_filter_.reset();
    }

    void setTau(float tau) {
        x1_filter_.setTau(tau);
        y1_filter_.setTau(tau);
        x2_filter_.setTau(tau);
        y2_filter_.setTau(tau);
    }

private:
    TimeBasedEMAFilter x1_filter_;
    TimeBasedEMAFilter y1_filter_;
    TimeBasedEMAFilter x2_filter_;
    TimeBasedEMAFilter y2_filter_;
};

// ============================================================================
// Unified Landmarks Smoothing Filter (supports multiple filter types)
// ============================================================================

/**
 * Unified Landmarks Smoothing Filter
 * Supports switching between different filter types via configuration
 */
class UnifiedLandmarksSmoothingFilter {
public:
    explicit UnifiedLandmarksSmoothingFilter(int num_landmarks = 478)
        : num_landmarks_(num_landmarks)
        , current_type_(SmoothingFilterType::TIME_BASED_EMA) {
        // Initialize with default Time-Based EMA
        time_ema_filter_ = std::make_unique<TimeBasedLandmarksSmoothingFilter>(num_landmarks, 0.08f);
    }

    void setFilterType(SmoothingFilterType type) {
        if (type == current_type_) return;
        current_type_ = type;
        resetAll();
    }

    void configure(const FilterConfig& config) {
        config_ = config;
        current_type_ = config.type;

        // Reinitialize filters with new parameters
        switch (config.type) {
            case SmoothingFilterType::ONE_EURO:
                one_euro_filter_ = std::make_unique<LandmarksSmoothingFilter>(
                    num_landmarks_,
                    config.one_euro_min_cutoff,
                    config.one_euro_beta,
                    config.one_euro_derivate_cutoff
                );
                break;
            case SmoothingFilterType::TIME_BASED_EMA:
                time_ema_filter_ = std::make_unique<TimeBasedLandmarksSmoothingFilter>(
                    num_landmarks_,
                    config.time_ema_tau
                );
                break;
            default:
                break;
        }
    }

    template<typename LandmarkType>
    void apply(std::vector<LandmarkType>& landmarks, double timestamp = -1.0) {
        switch (current_type_) {
            case SmoothingFilterType::ONE_EURO:
                if (one_euro_filter_) {
                    one_euro_filter_->apply(landmarks, timestamp);
                }
                break;
            case SmoothingFilterType::TIME_BASED_EMA:
                if (time_ema_filter_) {
                    time_ema_filter_->apply(landmarks, timestamp);
                }
                break;
            case SmoothingFilterType::NONE:
            default:
                // No smoothing
                break;
        }
    }

    void reset() {
        if (one_euro_filter_) one_euro_filter_->reset();
        if (time_ema_filter_) time_ema_filter_->reset();
    }

    [[nodiscard]] SmoothingFilterType getFilterType() const { return current_type_; }

private:
    void resetAll() {
        if (one_euro_filter_) one_euro_filter_->reset();
        if (time_ema_filter_) time_ema_filter_->reset();
    }

    int num_landmarks_;
    SmoothingFilterType current_type_;
    FilterConfig config_;

    std::unique_ptr<LandmarksSmoothingFilter> one_euro_filter_;
    std::unique_ptr<TimeBasedLandmarksSmoothingFilter> time_ema_filter_;
};

// ============================================================================
// Unified Detection Smoothing Filter (supports multiple filter types)
// ============================================================================

/**
 * Unified Detection Smoothing Filter
 * Supports switching between different filter types via configuration
 */
class UnifiedDetectionSmoothingFilter {
public:
    UnifiedDetectionSmoothingFilter()
        : current_type_(SmoothingFilterType::TIME_BASED_EMA) {
        // Initialize with default Time-Based EMA
        time_ema_filter_ = std::make_unique<TimeBasedDetectionSmoothingFilter>(0.06f);
    }

    void setFilterType(SmoothingFilterType type) {
        if (type == current_type_) return;
        current_type_ = type;
        resetAll();
    }

    void configure(const FilterConfig& config) {
        config_ = config;
        current_type_ = config.type;

        // Reinitialize filters with new parameters
        switch (config.type) {
            case SmoothingFilterType::ONE_EURO:
                one_euro_filter_ = std::make_unique<DetectionSmoothingFilter>(
                    config.one_euro_min_cutoff,
                    config.one_euro_beta,
                    config.one_euro_derivate_cutoff
                );
                break;
            case SmoothingFilterType::TIME_BASED_EMA:
                time_ema_filter_ = std::make_unique<TimeBasedDetectionSmoothingFilter>(
                    config.time_ema_tau
                );
                break;
            default:
                break;
        }
    }

    void apply(float& x1, float& y1, float& x2, float& y2, double timestamp = -1.0) {
        switch (current_type_) {
            case SmoothingFilterType::ONE_EURO:
                if (one_euro_filter_) {
                    one_euro_filter_->apply(x1, y1, x2, y2, timestamp);
                }
                break;
            case SmoothingFilterType::TIME_BASED_EMA:
                if (time_ema_filter_) {
                    time_ema_filter_->apply(x1, y1, x2, y2, timestamp);
                }
                break;
            case SmoothingFilterType::NONE:
            default:
                // No smoothing
                break;
        }
    }

    void reset() {
        if (one_euro_filter_) one_euro_filter_->reset();
        if (time_ema_filter_) time_ema_filter_->reset();
    }

    [[nodiscard]] SmoothingFilterType getFilterType() const { return current_type_; }

private:
    void resetAll() {
        if (one_euro_filter_) one_euro_filter_->reset();
        if (time_ema_filter_) time_ema_filter_->reset();
    }

    SmoothingFilterType current_type_;
    FilterConfig config_;

    std::unique_ptr<DetectionSmoothingFilter> one_euro_filter_;
    std::unique_ptr<TimeBasedDetectionSmoothingFilter> time_ema_filter_;
};

}  // namespace mnncv

#endif  // FACE_PIPELINE_FILTER_H
