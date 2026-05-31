/**
 * procrustes_solver_nodep.h
 *
 * Procrustes solver without Eigen dependency.
 * Used to find optimal rotation and scale transformation
 * between two point sets.
 *
 * Based on MediaPipe's procrustes_solver.cc
 */

#ifndef MNN_FACE_GEOMETRY_PROCRUSTES_SOLVER_NODEP_H
#define MNN_FACE_GEOMETRY_PROCRUSTES_SOLVER_NODEP_H

#include "mini_linalg.h"
#include <cmath>
#include <string>

namespace MNN::FaceGeometry {

/**
 * Status codes for procrustes operations
 */
enum class ProcrustesStatus {
    OK,
    INVALID_INPUT,
    NUMERICAL_ERROR
};

/**
 * Procrustes solver that finds the optimal rotation, scale, and translation
 * to align source points to target points.
 *
 * Solves the weighted orthogonal Procrustes problem:
 *   minimize || W (R * source + t - target) ||
 * where R is rotation-and-scale matrix, t is translation, W is weight matrix
 */
class ProcrustesSolver {
public:
    static constexpr float kAbsoluteErrorEps = 1e-9f;

    /**
     * Solve the weighted orthogonal Procrustes problem.
     *
     * @param source_points  3xN matrix of source points
     * @param target_points  3xN matrix of target points
     * @param point_weights  N-dimensional weight vector
     * @param transform_mat  Output 4x4 transformation matrix
     * @return Status code
     */
    static ProcrustesStatus SolveWeightedOrthogonalProblem(
        const Matrix3Xf& source_points,
        const Matrix3Xf& target_points,
        const VectorXf& point_weights,
        Matrix4f& transform_mat);

private:
    /**
     * Validate input points have same number of columns
     */
    static ProcrustesStatus ValidateInputPoints(
        const Matrix3Xf& source_points,
        const Matrix3Xf& target_points);

    /**
     * Validate point weights are non-negative and sum to positive value
     */
    static ProcrustesStatus ValidatePointWeights(
        int num_points,
        const VectorXf& point_weights);

    /**
     * Extract element-wise square root of weights
     */
    static VectorXf ExtractSquareRoot(const VectorXf& point_weights);

    /**
     * Combine 3x3 rotation-and-scale matrix with translation into 4x4 transform
     */
    static Matrix4f CombineTransformMatrix(
        const Matrix3f& r_and_s,
        const Vector3f& t);

    /**
     * Compute optimal rotation using SVD
     */
    static ProcrustesStatus ComputeOptimalRotation(
        const Matrix3f& design_matrix,
        Matrix3f& rotation);

    /**
     * Compute optimal scale factor
     */
    static ProcrustesStatus ComputeOptimalScale(
        const Matrix3Xf& centered_weighted_sources,
        const Matrix3Xf& weighted_sources,
        const Matrix3Xf& weighted_targets,
        const Matrix3f& rotation,
        float& scale);

    /**
     * Internal solver implementation
     */
    static ProcrustesStatus InternalSolveWeightedOrthogonalProblem(
        const Matrix3Xf& sources,
        const Matrix3Xf& targets,
        const VectorXf& sqrt_weights,
        Matrix4f& transform_mat);
};

//=============================================================================
// Implementation
//=============================================================================

inline ProcrustesStatus ProcrustesSolver::ValidateInputPoints(
    const Matrix3Xf& source_points,
    const Matrix3Xf& target_points) {

    if (source_points.cols() <= 0) {
        return ProcrustesStatus::INVALID_INPUT;
    }
    if (source_points.cols() != target_points.cols()) {
        return ProcrustesStatus::INVALID_INPUT;
    }
    return ProcrustesStatus::OK;
}

inline ProcrustesStatus ProcrustesSolver::ValidatePointWeights(
    int num_points,
    const VectorXf& point_weights) {

    if (point_weights.size() <= 0) {
        return ProcrustesStatus::INVALID_INPUT;
    }
    if (point_weights.size() != num_points) {
        return ProcrustesStatus::INVALID_INPUT;
    }

    float total_weight = 0.0f;
    for (int i = 0; i < num_points; ++i) {
        if (point_weights(i) < 0.0f) {
            return ProcrustesStatus::INVALID_INPUT;
        }
        total_weight += point_weights(i);
    }

    if (total_weight <= kAbsoluteErrorEps) {
        return ProcrustesStatus::INVALID_INPUT;
    }

    return ProcrustesStatus::OK;
}

inline VectorXf ProcrustesSolver::ExtractSquareRoot(const VectorXf& point_weights) {
    VectorXf sqrt_weights(point_weights.size());
    for (int i = 0; i < sqrt_weights.size(); ++i) {
        sqrt_weights(i) = std::sqrt(point_weights(i));
    }
    return sqrt_weights;
}

inline Matrix4f ProcrustesSolver::CombineTransformMatrix(
    const Matrix3f& r_and_s,
    const Vector3f& t) {

    Matrix4f result = Matrix4f::Identity();
    result.setTopLeftCorner3x3(r_and_s);
    result.setTranslation(t);
    return result;
}

inline ProcrustesStatus ProcrustesSolver::ComputeOptimalRotation(
    const Matrix3f& design_matrix,
    Matrix3f& rotation) {

    if (design_matrix.norm() <= kAbsoluteErrorEps) {
        return ProcrustesStatus::NUMERICAL_ERROR;
    }

    // SVD decomposition: design_matrix = U * S * V^T
    SVDResult3x3 svd = computeSVD3x3(design_matrix);

    Matrix3f postrotation = svd.U;
    Matrix3f prerotation = svd.V.transpose();

    // Ensure det(rotation) = +1 (not -1) to avoid reflection
    // If determinant is negative, flip sign of least singular value column
    if (postrotation.determinant() * prerotation.determinant() < 0.0f) {
        postrotation.scaleCol(2, -1.0f);
    }

    // rotation = U * V^T
    rotation = postrotation * prerotation;
    return ProcrustesStatus::OK;
}

inline ProcrustesStatus ProcrustesSolver::ComputeOptimalScale(
    const Matrix3Xf& centered_weighted_sources,
    const Matrix3Xf& weighted_sources,
    const Matrix3Xf& weighted_targets,
    const Matrix3f& rotation,
    float& scale) {

    // rotated_sources = rotation * centered_weighted_sources
    Matrix3Xf rotated = centered_weighted_sources.leftMultiply(rotation);

    // numerator = trace(rotated^T * weighted_targets)
    //           = sum of element-wise products
    float numerator = rotated.cwiseProductSum(weighted_targets);

    // denominator = trace(centered^T * sources)
    float denominator = centered_weighted_sources.cwiseProductSum(weighted_sources);

    if (denominator <= kAbsoluteErrorEps) {
        return ProcrustesStatus::NUMERICAL_ERROR;
    }

    scale = numerator / denominator;

    if (scale <= kAbsoluteErrorEps) {
        return ProcrustesStatus::NUMERICAL_ERROR;
    }

    return ProcrustesStatus::OK;
}

inline ProcrustesStatus ProcrustesSolver::InternalSolveWeightedOrthogonalProblem(
    const Matrix3Xf& sources,
    const Matrix3Xf& targets,
    const VectorXf& sqrt_weights,
    Matrix4f& transform_mat) {

    // Apply weights to points (element-wise row multiplication)
    Matrix3Xf weighted_sources = sources.rowwiseProduct(sqrt_weights);
    Matrix3Xf weighted_targets = targets.rowwiseProduct(sqrt_weights);

    // Total weight: w = sum(sqrt_weights^2)
    float total_weight = sqrt_weights.cwiseProduct(sqrt_weights).sum();

    // Compute center of mass for sources
    // twice_weighted_sources = weighted_sources * diag(sqrt_weights)
    Matrix3Xf twice_weighted_sources = weighted_sources.rowwiseProduct(sqrt_weights);
    Vector3f source_center_of_mass = twice_weighted_sources.rowwiseSum() * (1.0f / total_weight);

    // centered_weighted_sources = weighted_sources - source_center * sqrt_weights^T
    Matrix3Xf centered_weighted_sources(sources.cols());
    for (int j = 0; j < sources.cols(); ++j) {
        centered_weighted_sources(0, j) = weighted_sources(0, j) - source_center_of_mass(0) * sqrt_weights(j);
        centered_weighted_sources(1, j) = weighted_sources(1, j) - source_center_of_mass(1) * sqrt_weights(j);
        centered_weighted_sources(2, j) = weighted_sources(2, j) - source_center_of_mass(2) * sqrt_weights(j);
    }

    // Compute design matrix for rotation: weighted_targets * centered_weighted_sources^T
    Matrix3f design_matrix = weighted_targets.outerProduct(centered_weighted_sources);

    // Compute optimal rotation
    Matrix3f rotation;
    ProcrustesStatus status = ComputeOptimalRotation(design_matrix, rotation);
    if (status != ProcrustesStatus::OK) {
        return status;
    }

    // Compute optimal scale
    float scale;
    status = ComputeOptimalScale(centered_weighted_sources, weighted_sources,
                                  weighted_targets, rotation, scale);
    if (status != ProcrustesStatus::OK) {
        return status;
    }

    // rotation_and_scale = scale * rotation
    Matrix3f rotation_and_scale = rotation * scale;

    // Compute optimal translation
    // diff = weighted_targets - rotation_and_scale * weighted_sources
    Matrix3Xf rotated_sources = weighted_sources.leftMultiply(rotation_and_scale);
    Matrix3Xf pointwise_diffs = weighted_targets - rotated_sources;

    // Weight the diffs again and sum
    Matrix3Xf weighted_diffs = pointwise_diffs.rowwiseProduct(sqrt_weights);
    Vector3f translation = weighted_diffs.rowwiseSum() * (1.0f / total_weight);

    // Combine into 4x4 transform matrix
    transform_mat = CombineTransformMatrix(rotation_and_scale, translation);

    return ProcrustesStatus::OK;
}

inline ProcrustesStatus ProcrustesSolver::SolveWeightedOrthogonalProblem(
    const Matrix3Xf& source_points,
    const Matrix3Xf& target_points,
    const VectorXf& point_weights,
    Matrix4f& transform_mat) {

    // Validate inputs
    ProcrustesStatus status = ValidateInputPoints(source_points, target_points);
    if (status != ProcrustesStatus::OK) {
        return status;
    }

    status = ValidatePointWeights(source_points.cols(), point_weights);
    if (status != ProcrustesStatus::OK) {
        return status;
    }

    // Extract square root from point weights
    VectorXf sqrt_weights = ExtractSquareRoot(point_weights);

    // Solve the problem
    return InternalSolveWeightedOrthogonalProblem(
        source_points, target_points, sqrt_weights, transform_mat);
}

} // namespace MNN::FaceGeometry


#endif  // MNN_FACE_GEOMETRY_PROCRUSTES_SOLVER_NODEP_H

