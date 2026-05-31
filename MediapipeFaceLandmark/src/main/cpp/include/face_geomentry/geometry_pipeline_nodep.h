/**
 * geometry_pipeline_nodep.h
 *
 * Face geometry pipeline without Eigen dependency.
 * Converts 2D face landmarks to 3D metric space and estimates pose.
 *
 * Based on MediaPipe's geometry_pipeline.cc
 */

#ifndef MNN_FACE_GEOMETRY_GEOMETRY_PIPELINE_NODEP_H
#define MNN_FACE_GEOMETRY_GEOMETRY_PIPELINE_NODEP_H

#include "mini_linalg.h"
#include "procrustes_solver_nodep.h"
#include "canonical_face_mesh_data.h"
#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <fstream>
#include <cstdlib>

namespace MNN::FaceGeometry {

//=============================================================================
// Data structures
//=============================================================================

/**
 * Perspective camera frustum parameters
 */
struct PerspectiveCameraFrustum {
    float left;
    float right;
    float bottom;
    float top;
    float near;
    float far;

    PerspectiveCameraFrustum(float vertical_fov_degrees, float near_val, float far_val,
                             int frame_width, int frame_height) {
        static constexpr float kDegreesToRadians = 3.14159265358979323846f / 180.0f;

        float height_at_near = 2.0f * near_val *
            std::tan(0.5f * kDegreesToRadians * vertical_fov_degrees);
        float width_at_near = static_cast<float>(frame_width) * height_at_near /
            static_cast<float>(frame_height);

        left = -0.5f * width_at_near;
        right = 0.5f * width_at_near;
        bottom = -0.5f * height_at_near;
        top = 0.5f * height_at_near;
        near = near_val;
        far = far_val;
    }
};

/**
 * Environment configuration
 */
struct Environment {
    // Origin point location: TOP_LEFT_CORNER or BOTTOM_LEFT_CORNER
    bool origin_is_top_left = true;

    // Perspective camera parameters
    float vertical_fov_degrees = 63.0f;
    float near = 1.0f;      // 1cm
    float far = 10000.0f;   // 100m
};

/**
 * Input source type
 */
enum class InputSource {
    FACE_DETECTION_PIPELINE,
    FACE_LANDMARK_PIPELINE
};

/**
 * 3D landmark with x, y, z coordinates
 */
struct Landmark3D {
    float x, y, z;
};

/**
 * Normalized 2D landmark (0-1 range)
 */
struct NormalizedLandmark {
    float x, y, z;  // z is relative depth
};

/**
 * Face geometry result
 */
struct FaceGeometry {
    std::vector<Landmark3D> metric_landmarks;
    Matrix4f pose_transform_matrix;
};

//=============================================================================
// Screen to Metric Space Converter
//=============================================================================

class ScreenToMetricSpaceConverter {
public:
    ScreenToMetricSpaceConverter(
        bool origin_is_top_left,
        InputSource input_source,
        const Matrix3Xf& canonical_metric_landmarks,
        const VectorXf& landmark_weights)
        : origin_is_top_left_(origin_is_top_left),
          input_source_(input_source),
          canonical_metric_landmarks_(canonical_metric_landmarks),
          landmark_weights_(landmark_weights) {}

    /**
     * Convert screen landmarks to metric landmarks and estimate pose.
     *
     * Algorithm summary:
     * (1) Project X- and Y- screen landmark coordinates at the Z near plane.
     * (2) Estimate canonical-to-runtime scale using Procrustes solver.
     * (3) Unproject screen landmarks using the scale.
     * (4) Re-estimate scale with unprojected landmarks.
     * (5) Final unprojection with combined scale.
     * (6) Align runtime landmarks with canonical landmarks.
     *
     * @param screen_landmarks  Input 2D landmarks (normalized 0-1)
     * @param pcf               Perspective camera frustum
     * @param metric_landmarks  Output 3D landmarks in metric space
     * @param pose_transform    Output 4x4 pose transformation matrix
     * @return true on success
     */
    bool Convert(
        const std::vector<NormalizedLandmark>& screen_landmarks,
        const PerspectiveCameraFrustum& pcf,
        std::vector<Landmark3D>& metric_landmarks,
        Matrix4f& pose_transform);

private:
    void ProjectXY(const PerspectiveCameraFrustum& pcf, Matrix3Xf& landmarks);
    float EstimateScale(const Matrix3Xf& landmarks);
    void MoveAndRescaleZ(const PerspectiveCameraFrustum& pcf,
                         float depth_offset, float scale, Matrix3Xf& landmarks);
    void UnprojectXY(const PerspectiveCameraFrustum& pcf, Matrix3Xf& landmarks);
    void ChangeHandedness(Matrix3Xf& landmarks);

    // Convert landmark list to matrix
    static Matrix3Xf ToMatrix(const std::vector<NormalizedLandmark>& landmarks);

    // Convert matrix to landmark list
    static std::vector<Landmark3D> ToLandmarkList(const Matrix3Xf& matrix);

    bool origin_is_top_left_;
    InputSource input_source_;
    Matrix3Xf canonical_metric_landmarks_;
    VectorXf landmark_weights_;
};

//=============================================================================
// ScreenToMetricSpaceConverter Implementation
//=============================================================================

inline Matrix3Xf ScreenToMetricSpaceConverter::ToMatrix(
    const std::vector<NormalizedLandmark>& landmarks) {

    Matrix3Xf result(static_cast<int>(landmarks.size()));
    for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
        result(0, i) = landmarks[i].x;
        result(1, i) = landmarks[i].y;
        result(2, i) = landmarks[i].z;
    }
    return result;
}

inline std::vector<Landmark3D> ScreenToMetricSpaceConverter::ToLandmarkList(
    const Matrix3Xf& matrix) {

    std::vector<Landmark3D> result(matrix.cols());
    for (int i = 0; i < matrix.cols(); ++i) {
        result[i].x = matrix(0, i);
        result[i].y = matrix(1, i);
        result[i].z = matrix(2, i);
    }
    return result;
}

inline void ScreenToMetricSpaceConverter::ProjectXY(
    const PerspectiveCameraFrustum& pcf,
    Matrix3Xf& landmarks) {

    float x_scale = pcf.right - pcf.left;
    float y_scale = pcf.top - pcf.bottom;
    float x_translation = pcf.left;
    float y_translation = pcf.bottom;

    for (int j = 0; j < landmarks.cols(); ++j) {
        float y = landmarks(1, j);
        if (origin_is_top_left_) {
            y = 1.0f - y;
        }

        landmarks(0, j) = landmarks(0, j) * x_scale + x_translation;
        landmarks(1, j) = y * y_scale + y_translation;
        landmarks(2, j) = landmarks(2, j) * x_scale;  // z uses x_scale
    }
}

inline float ScreenToMetricSpaceConverter::EstimateScale(const Matrix3Xf& landmarks) {
    Matrix4f transform_mat;
    ProcrustesStatus status = ProcrustesSolver::SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, landmarks, landmark_weights_, transform_mat);

    if (status != ProcrustesStatus::OK) {
        return 1.0f;  // Fallback
    }

    // Scale is the norm of the first column of the 3x3 rotation part
    return transform_mat.colNorm(0);
}

inline void ScreenToMetricSpaceConverter::MoveAndRescaleZ(
    const PerspectiveCameraFrustum& pcf,
    float depth_offset,
    float scale,
    Matrix3Xf& landmarks) {

    for (int j = 0; j < landmarks.cols(); ++j) {
        landmarks(2, j) = (landmarks(2, j) - depth_offset + pcf.near) / scale;
    }
}

inline void ScreenToMetricSpaceConverter::UnprojectXY(
    const PerspectiveCameraFrustum& pcf,
    Matrix3Xf& landmarks) {

    float inv_near = 1.0f / pcf.near;
    for (int j = 0; j < landmarks.cols(); ++j) {
        float z = landmarks(2, j);
        landmarks(0, j) = landmarks(0, j) * z * inv_near;
        landmarks(1, j) = landmarks(1, j) * z * inv_near;
    }
}

inline void ScreenToMetricSpaceConverter::ChangeHandedness(Matrix3Xf& landmarks) {
    landmarks.negateRow(2);  // Negate z
}

inline bool ScreenToMetricSpaceConverter::Convert(
    const std::vector<NormalizedLandmark>& screen_landmarks,
    const PerspectiveCameraFrustum& pcf,
    std::vector<Landmark3D>& metric_landmarks,
    Matrix4f& pose_transform) {

    if (screen_landmarks.size() != static_cast<size_t>(canonical_metric_landmarks_.cols())) {
        return false;
    }

    Matrix3Xf landmarks = ToMatrix(screen_landmarks);

    ProjectXY(pcf, landmarks);

    // Compute depth offset (mean of z values)
    float depth_offset = 0.0f;
    for (int j = 0; j < landmarks.cols(); ++j) {
        depth_offset += landmarks(2, j);
    }
    depth_offset /= landmarks.cols();

    // First iteration: estimate scale without unprojecting
    Matrix3Xf intermediate = landmarks;
    ChangeHandedness(intermediate);

    float first_scale = EstimateScale(intermediate);

    // Second iteration: unproject and re-estimate
    intermediate = landmarks;
    MoveAndRescaleZ(pcf, depth_offset, first_scale, intermediate);
    UnprojectXY(pcf, intermediate);
    ChangeHandedness(intermediate);

    // For face detection input, rewrite Z from canonical landmarks
    if (input_source_ == InputSource::FACE_DETECTION_PIPELINE) {
        Matrix4f intermediate_transform;
        ProcrustesSolver::SolveWeightedOrthogonalProblem(
            canonical_metric_landmarks_, intermediate, landmark_weights_, intermediate_transform);

        Matrix3Xf transformed = applyTransform(intermediate_transform, canonical_metric_landmarks_);
        for (int j = 0; j < intermediate.cols(); ++j) {
            intermediate(2, j) = transformed(2, j);
        }
    }

    float second_scale = EstimateScale(intermediate);
    float total_scale = first_scale * second_scale;

    // Final unprojection
    MoveAndRescaleZ(pcf, depth_offset, total_scale, landmarks);
    UnprojectXY(pcf, landmarks);
    ChangeHandedness(landmarks);

    // Now landmarks are in metric space
    Matrix3Xf& metric = landmarks;

    // Solve for pose transform
    ProcrustesStatus status = ProcrustesSolver::SolveWeightedOrthogonalProblem(
        canonical_metric_landmarks_, metric, landmark_weights_, pose_transform);

    if (status != ProcrustesStatus::OK) {
        return false;
    }

    // For face detection, rewrite Z and re-solve
    if (input_source_ == InputSource::FACE_DETECTION_PIPELINE) {
        Matrix3Xf transformed = applyTransform(pose_transform, canonical_metric_landmarks_);
        for (int j = 0; j < metric.cols(); ++j) {
            metric(2, j) = transformed(2, j);
        }

        status = ProcrustesSolver::SolveWeightedOrthogonalProblem(
            canonical_metric_landmarks_, metric, landmark_weights_, pose_transform);

        if (status != ProcrustesStatus::OK) {
            return false;
        }
    }

    // Align metric landmarks with canonical landmarks
    Matrix4f inverse_transform = pose_transform.inverse();
    metric = applyTransform(inverse_transform, metric);

    metric_landmarks = ToLandmarkList(metric);
    return true;
}

//=============================================================================
// Face Geometry Pipeline
//=============================================================================

class GeometryPipeline {
public:
    /**
     * Create a geometry pipeline.
     *
     * @param environment       Environment configuration
     * @param input_source      Input source type
     * @param canonical_mesh    3xN matrix of canonical face mesh vertices
     * @param landmark_weights  Weights for Procrustes alignment
     * @return Unique pointer to pipeline, or nullptr on error
     */
    static std::unique_ptr<GeometryPipeline> Create(
        const Environment& environment,
        InputSource input_source,
        const Matrix3Xf& canonical_mesh,
        const VectorXf& landmark_weights);

    /**
     * Estimate face geometry from landmarks.
     *
     * @param face_landmarks    Vector of normalized landmarks for each face
     * @param frame_width       Image width in pixels
     * @param frame_height      Image height in pixels
     * @param results           Output face geometry results
     * @return true on success
     */
    bool EstimateFaceGeometry(
        const std::vector<std::vector<NormalizedLandmark>>& face_landmarks,
        int frame_width,
        int frame_height,
        std::vector<FaceGeometry>& results);

private:
    GeometryPipeline(const Environment& env,
                     std::unique_ptr<ScreenToMetricSpaceConverter> converter)
        : environment_(env), converter_(std::move(converter)) {}

    static bool IsLandmarkListTooCompact(const std::vector<NormalizedLandmark>& landmarks);

    Environment environment_;
    std::unique_ptr<ScreenToMetricSpaceConverter> converter_;
};

//=============================================================================
// GeometryPipeline Implementation
//=============================================================================

inline std::unique_ptr<GeometryPipeline> GeometryPipeline::Create(
    const Environment& environment,
    InputSource input_source,
    const Matrix3Xf& canonical_mesh,
    const VectorXf& landmark_weights) {

    if (canonical_mesh.cols() <= 0 ||
        landmark_weights.size() != canonical_mesh.cols()) {
        return nullptr;
    }

    auto converter = std::make_unique<ScreenToMetricSpaceConverter>(
        environment.origin_is_top_left,
        input_source,
        canonical_mesh,
        landmark_weights);

    return std::unique_ptr<GeometryPipeline>(
        new GeometryPipeline(environment, std::move(converter)));
}

inline bool GeometryPipeline::IsLandmarkListTooCompact(
    const std::vector<NormalizedLandmark>& landmarks) {

    if (landmarks.empty()) return true;

    // Compute mean
    float mean_x = 0.0f, mean_y = 0.0f;
    for (const auto& lm : landmarks) {
        mean_x += lm.x;
        mean_y += lm.y;
    }
    mean_x /= landmarks.size();
    mean_y /= landmarks.size();

    // Find max distance from mean
    float max_sq_dist = 0.0f;
    for (const auto& lm : landmarks) {
        float dx = lm.x - mean_x;
        float dy = lm.y - mean_y;
        max_sq_dist = std::max(max_sq_dist, dx * dx + dy * dy);
    }

    static constexpr float kThreshold = 1e-3f;
    return std::sqrt(max_sq_dist) <= kThreshold;
}

inline bool GeometryPipeline::EstimateFaceGeometry(
    const std::vector<std::vector<NormalizedLandmark>>& face_landmarks,
    int frame_width,
    int frame_height,
    std::vector<FaceGeometry>& results) {

    if (frame_width <= 0 || frame_height <= 0) {
        return false;
    }

    PerspectiveCameraFrustum pcf(
        environment_.vertical_fov_degrees,
        environment_.near,
        environment_.far,
        frame_width,
        frame_height);

    results.clear();

    for (const auto& landmarks : face_landmarks) {
        // Skip faces that are too compact (numerically unstable)
        if (IsLandmarkListTooCompact(landmarks)) {
            continue;
        }

        FaceGeometry geometry;
        bool success = converter_->Convert(
            landmarks, pcf, geometry.metric_landmarks, geometry.pose_transform_matrix);

        if (success) {
            results.push_back(std::move(geometry));
        }
    }

    return true;
}

//=============================================================================
// Utility: Load canonical mesh data from binary data
//=============================================================================

/**
 * Structure to hold geometry pipeline metadata
 */
struct GeometryPipelineMetadata {
    Matrix3Xf canonical_mesh;      // 3xN vertex positions
    VectorXf landmark_weights;     // N weights for Procrustes
    InputSource input_source = InputSource::FACE_LANDMARK_PIPELINE;
};

/**
 * Create a default 468-point face mesh with typical weights.
 * This is a simplified version - in production, load from metadata file.
 */
inline GeometryPipelineMetadata CreateDefaultFaceMeshMetadata(int num_landmarks = 468) {
    GeometryPipelineMetadata metadata;
    metadata.canonical_mesh.resize(3, num_landmarks);
    metadata.landmark_weights.resize(num_landmarks);

    // Initialize with zeros - in production, load actual data
    for (int i = 0; i < num_landmarks; ++i) {
        metadata.canonical_mesh(0, i) = 0.0f;
        metadata.canonical_mesh(1, i) = 0.0f;
        metadata.canonical_mesh(2, i) = 0.0f;
        metadata.landmark_weights(i) = 0.0f;
    }

    // Typical Procrustes landmark basis (key points with weights)
    // These are the most stable points for alignment
    struct WeightedLandmark { int id; float weight; };
    std::vector<WeightedLandmark> basis = {
        {4, 0.070909939706326f},
        {6, 0.032100144773722f},
        {10, 0.008446550928056f},
        {33, 0.058724168688059f},
        {54, 0.007667080033571f},
        {67, 0.009078059345484f},
        {117, 0.009791937656701f},
        {119, 0.014565368182957f},
        {121, 0.018591361120343f},
        {127, 0.005197994410992f},
        {129, 0.120625205338001f},
        {132, 0.005560018587857f},
        {133, 0.053286183625460f},
        {136, 0.066890455782413f},
        {143, 0.014816547743976f},
        {147, 0.014262833632529f},
        {198, 0.025462191551924f},
        {205, 0.047252278774977f},
        {263, 0.058724168688059f},
        {284, 0.007667080033571f},
        {297, 0.009078059345484f},
        {346, 0.009791937656701f},
        {348, 0.014565368182957f},
        {350, 0.018591361120343f},
        {356, 0.005197994410992f},
        {358, 0.120625205338001f},
        {361, 0.005560018587857f},
        {362, 0.053286183625460f},
        {365, 0.066890455782413f},
        {372, 0.014816547743976f},
        {376, 0.014262833632529f},
        {420, 0.025462191551924f},
        {425, 0.047252278774977f}
    };

    for (const auto& wl : basis) {
        if (wl.id < num_landmarks) {
            metadata.landmark_weights(wl.id) = wl.weight;
        }
    }

    return metadata;
}

/**
 * Create face mesh metadata from hardcoded canonical data.
 * This is the RECOMMENDED method for production use - no file I/O needed.
 *
 * Uses pre-computed data from MediaPipe's geometry_pipeline_metadata_landmarks.pbtxt
 * embedded in canonical_face_mesh_data.h
 *
 * @return GeometryPipelineMetadata with 468 vertices and Procrustes weights
 */
inline GeometryPipelineMetadata CreateCanonicalFaceMeshMetadata() {
    GeometryPipelineMetadata metadata;
    metadata.input_source = InputSource::FACE_LANDMARK_PIPELINE;

    // Allocate mesh and weights
    metadata.canonical_mesh.resize(3, kNumFaceMeshLandmarks);
    metadata.landmark_weights.resize(kNumFaceMeshLandmarks);

    // Copy vertex positions from hardcoded data
    for (int i = 0; i < kNumFaceMeshLandmarks; ++i) {
        metadata.canonical_mesh(0, i) = kCanonicalFaceMeshVertices[i * 3 + 0];  // x
        metadata.canonical_mesh(1, i) = kCanonicalFaceMeshVertices[i * 3 + 1];  // y
        metadata.canonical_mesh(2, i) = kCanonicalFaceMeshVertices[i * 3 + 2];  // z
        metadata.landmark_weights(i) = 0.0f;  // Initialize to zero
    }

    // Set Procrustes landmark weights
    for (int i = 0; i < kNumProcrustesLandmarkBasis; ++i) {
        int id = kProcrustesLandmarkBasis[i].id;
        float weight = kProcrustesLandmarkBasis[i].weight;
        if (id < kNumFaceMeshLandmarks) {
            metadata.landmark_weights(id) = weight;
        }
    }

    return metadata;
}

/**
 * Load geometry pipeline metadata from a pbtxt file.
 * Parses the MediaPipe geometry_pipeline_metadata format.
 *
 * NOTE: For production Android/iOS use, prefer CreateCanonicalFaceMeshMetadata()
 * which uses hardcoded data and requires no file I/O.
 *
 * @param filepath  Path to the pbtxt file
 * @return GeometryPipelineMetadata struct (empty if parsing failed)
 */
inline GeometryPipelineMetadata LoadMetadataFromPbtxt(const std::string& filepath) {
    GeometryPipelineMetadata metadata;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        return metadata;
    }

    std::vector<float> vertex_buffer;
    std::vector<std::pair<int, float>> landmark_basis;

    std::string line;
    bool in_canonical_mesh = false;

    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Parse input_source
        if (line.find("input_source:") == 0) {
            if (line.find("FACE_DETECTION_PIPELINE") != std::string::npos) {
                metadata.input_source = InputSource::FACE_DETECTION_PIPELINE;
            } else {
                metadata.input_source = InputSource::FACE_LANDMARK_PIPELINE;
            }
        }
        // Parse procrustes_landmark_basis
        else if (line.find("procrustes_landmark_basis") == 0) {
            int id = -1;
            float weight = 0.0f;

            size_t id_pos = line.find("landmark_id:");
            size_t weight_pos = line.find("weight:");

            if (id_pos != std::string::npos) {
                id = std::atoi(line.c_str() + id_pos + 12);
            }
            if (weight_pos != std::string::npos) {
                weight = std::atof(line.c_str() + weight_pos + 7);
            }

            if (id >= 0) {
                landmark_basis.push_back({id, weight});
            }
        }
        // Parse canonical_mesh section
        else if (line.find("canonical_mesh:") == 0) {
            in_canonical_mesh = true;
        }
        else if (in_canonical_mesh && line.find("vertex_buffer:") == 0) {
            float value = std::atof(line.c_str() + 14);
            vertex_buffer.push_back(value);
        }
        else if (in_canonical_mesh && line == "}") {
            // End of canonical_mesh section if we have vertices
            if (!vertex_buffer.empty()) {
                in_canonical_mesh = false;
            }
        }
    }

    file.close();

    // vertex_buffer format is VERTEX_PT: x, y, z, u, v per vertex
    // We only need x, y, z
    int num_vertices = static_cast<int>(vertex_buffer.size()) / 5;
    if (num_vertices <= 0) {
        return metadata;
    }

    metadata.canonical_mesh.resize(3, num_vertices);
    metadata.landmark_weights.resize(num_vertices);

    // Initialize weights to zero
    for (int i = 0; i < num_vertices; ++i) {
        metadata.landmark_weights(i) = 0.0f;
    }

    // Extract xyz coordinates (skip uv)
    for (int i = 0; i < num_vertices; ++i) {
        metadata.canonical_mesh(0, i) = vertex_buffer[i * 5 + 0];  // x
        metadata.canonical_mesh(1, i) = vertex_buffer[i * 5 + 1];  // y
        metadata.canonical_mesh(2, i) = vertex_buffer[i * 5 + 2];  // z
    }

    // Set landmark weights
    for (const auto& lw : landmark_basis) {
        if (lw.first < num_vertices) {
            metadata.landmark_weights(lw.first) = lw.second;
        }
    }

    return metadata;
}

} // namespace MNN::FaceGeometry


#endif  // MNN_FACE_GEOMETRY_GEOMETRY_PIPELINE_NODEP_H

