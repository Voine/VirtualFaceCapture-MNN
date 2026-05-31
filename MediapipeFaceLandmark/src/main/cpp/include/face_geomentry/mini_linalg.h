/**
 * mini_linalg.h
 *
 * A minimal linear algebra library for Face Geometry.
 * This replaces Eigen dependency with a lightweight implementation.
 *
 * Implements only what Face Geometry needs:
 * - Matrix3f, Matrix4f (fixed size)
 * - Matrix3Xf (dynamic columns)
 * - VectorXf (dynamic size vector)
 * - Vector3f (fixed size)
 * - SVD decomposition for 3x3 matrices
 *
 */

#ifndef MNN_FACE_GEOMETRY_MINI_LINALG_H
#define MNN_FACE_GEOMETRY_MINI_LINALG_H

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include "neon_intrinsics.h"

namespace MNN::FaceGeometry {

// Forward declarations
class Matrix3f;
class Matrix4f;
class Matrix3Xf;
class VectorXf;
class Vector3f;

//=============================================================================
// Vector3f - Fixed size 3D vector
//=============================================================================
class Vector3f {
public:
    float data[3];

    Vector3f() { data[0] = data[1] = data[2] = 0.0f; }
    Vector3f(float x, float y, float z) { data[0] = x; data[1] = y; data[2] = z; }

    float& operator()(int i) { return data[i]; }
    float operator()(int i) const { return data[i]; }

    float& x() { return data[0]; }
    float& y() { return data[1]; }
    float& z() { return data[2]; }
    float x() const { return data[0]; }
    float y() const { return data[1]; }
    float z() const { return data[2]; }

    Vector3f operator+(const Vector3f& other) const {
        return Vector3f(data[0] + other.data[0],
                       data[1] + other.data[1],
                       data[2] + other.data[2]);
    }

    Vector3f operator-(const Vector3f& other) const {
        return Vector3f(data[0] - other.data[0],
                       data[1] - other.data[1],
                       data[2] - other.data[2]);
    }

    Vector3f operator*(float s) const {
        return Vector3f(data[0] * s, data[1] * s, data[2] * s);
    }

    float norm() const {
        return std::sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2]);
    }

    float squaredNorm() const {
        return data[0]*data[0] + data[1]*data[1] + data[2]*data[2];
    }
};

//=============================================================================
// VectorXf - Dynamic size vector
//=============================================================================
class VectorXf {
private:
    std::vector<float> data_;

public:
    VectorXf() = default;
    explicit VectorXf(int size) : data_(size, 0.0f) {}

    void resize(int size) { data_.resize(size, 0.0f); }
    int size() const { return static_cast<int>(data_.size()); }

    float& operator()(int i) { return data_[i]; }
    float operator()(int i) const { return data_[i]; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    static VectorXf Zero(int size) {
        VectorXf v(size);
        return v;
    }

    VectorXf cwiseProduct(const VectorXf& other) const {
        VectorXf result(size());
#if MNN_USE_NEON
        NEON::mul_neon(data_.data(), other.data(), result.data(), size());
#else
        for (int i = 0; i < size(); ++i) {
            result(i) = data_[i] * other(i);
        }
#endif
        return result;
    }

    float sum() const {
#if MNN_USE_NEON
        return NEON::sum_neon(data_.data(), size());
#else
        float s = 0.0f;
        for (int i = 0; i < size(); ++i) {
            s += data_[i];
        }
        return s;
#endif
    }

    VectorXf cwiseSqrt() const {
        VectorXf result(size());
        for (int i = 0; i < size(); ++i) {
            result(i) = std::sqrt(data_[i]);
        }
        return result;
    }
};

//=============================================================================
// Matrix3f - 3x3 fixed size matrix (column-major storage)
//=============================================================================
class Matrix3f {
public:
    float data[9];  // Column-major: data[col*3 + row]

    Matrix3f() { std::memset(data, 0, sizeof(data)); }

    static Matrix3f Identity() {
        Matrix3f m;
        m.data[0] = m.data[4] = m.data[8] = 1.0f;
        return m;
    }

    static Matrix3f Zero() {
        return Matrix3f();
    }

    float& operator()(int row, int col) { return data[col * 3 + row]; }
    float operator()(int row, int col) const { return data[col * 3 + row]; }

    Matrix3f transpose() const {
        Matrix3f result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    Matrix3f operator*(const Matrix3f& other) const {
        Matrix3f result;
#if MNN_USE_NEON
        NEON::mat3_mul_neon(data, other.data, result.data);
#else
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
#endif
        return result;
    }

    Vector3f operator*(const Vector3f& v) const {
        Vector3f result;
        for (int i = 0; i < 3; ++i) {
            result(i) = (*this)(i, 0) * v(0) + (*this)(i, 1) * v(1) + (*this)(i, 2) * v(2);
        }
        return result;
    }

    Matrix3f operator*(float s) const {
        Matrix3f result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = data[i] * s;
        }
        return result;
    }

    Matrix3f operator+(const Matrix3f& other) const {
        Matrix3f result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Matrix3f operator-(const Matrix3f& other) const {
        Matrix3f result;
        for (int i = 0; i < 9; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    // Column access
    Vector3f col(int j) const {
        return Vector3f((*this)(0, j), (*this)(1, j), (*this)(2, j));
    }

    void setCol(int j, const Vector3f& v) {
        (*this)(0, j) = v(0);
        (*this)(1, j) = v(1);
        (*this)(2, j) = v(2);
    }

    // Multiply column by scalar
    void scaleCol(int j, float s) {
        (*this)(0, j) *= s;
        (*this)(1, j) *= s;
        (*this)(2, j) *= s;
    }

    float determinant() const {
        return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
             - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
             + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
    }

    float norm() const {
        float sum = 0.0f;
        for (int i = 0; i < 9; ++i) {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }
};

//=============================================================================
// Matrix4f - 4x4 fixed size matrix (column-major storage)
//=============================================================================
class Matrix4f {
public:
    float data[16];  // Column-major: data[col*4 + row]

    Matrix4f() { std::memset(data, 0, sizeof(data)); }

    static Matrix4f Identity() {
        Matrix4f m;
        m.data[0] = m.data[5] = m.data[10] = m.data[15] = 1.0f;
        return m;
    }

    static Matrix4f Zero() {
        return Matrix4f();
    }

    float& operator()(int row, int col) { return data[col * 4 + row]; }
    float operator()(int row, int col) const { return data[col * 4 + row]; }

    Matrix4f transpose() const {
        Matrix4f result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result(i, j) = (*this)(j, i);
            }
        }
        return result;
    }

    Matrix4f operator*(const Matrix4f& other) const {
        Matrix4f result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    Matrix4f operator*(float s) const {
        Matrix4f result;
        for (int i = 0; i < 16; ++i) {
            result.data[i] = data[i] * s;
        }
        return result;
    }

    // Column norm (useful for extracting scale)
    float colNorm(int col) const {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float v = (*this)(i, col);
            sum += v * v;
        }
        return std::sqrt(sum);
    }

    // Get 3x3 top-left block
    Matrix3f topLeftCorner3x3() const {
        Matrix3f result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result(i, j) = (*this)(i, j);
            }
        }
        return result;
    }

    // Set 3x3 top-left block
    void setTopLeftCorner3x3(const Matrix3f& m) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                (*this)(i, j) = m(i, j);
            }
        }
    }

    // Set translation (column 3, rows 0-2)
    void setTranslation(const Vector3f& t) {
        (*this)(0, 3) = t(0);
        (*this)(1, 3) = t(1);
        (*this)(2, 3) = t(2);
    }

    // Matrix inverse using Gaussian elimination with partial pivoting
    Matrix4f inverse() const {
        Matrix4f result = Matrix4f::Identity();
        Matrix4f temp = *this;

        for (int col = 0; col < 4; ++col) {
            // Find pivot
            int maxRow = col;
            float maxVal = std::abs(temp(col, col));
            for (int row = col + 1; row < 4; ++row) {
                float val = std::abs(temp(row, col));
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col) {
                for (int j = 0; j < 4; ++j) {
                    std::swap(temp(col, j), temp(maxRow, j));
                    std::swap(result(col, j), result(maxRow, j));
                }
            }

            // Check for singular matrix
            float pivot = temp(col, col);
            if (std::abs(pivot) < std::numeric_limits<float>::epsilon()) {
                // Return identity for singular matrix
                return Matrix4f::Identity();
            }

            // Scale row
            float invPivot = 1.0f / pivot;
            for (int j = 0; j < 4; ++j) {
                temp(col, j) *= invPivot;
                result(col, j) *= invPivot;
            }

            // Eliminate column
            for (int row = 0; row < 4; ++row) {
                if (row != col) {
                    float factor = temp(row, col);
                    for (int j = 0; j < 4; ++j) {
                        temp(row, j) -= factor * temp(col, j);
                        result(row, j) -= factor * result(col, j);
                    }
                }
            }
        }

        return result;
    }
};

//=============================================================================
// Matrix3Xf - 3 rows, dynamic columns matrix (column-major storage)
//=============================================================================
class Matrix3Xf {
private:
    std::vector<float> data_;  // Column-major
    int cols_;

public:
    Matrix3Xf() : cols_(0) {}
    explicit Matrix3Xf(int cols) : data_(3 * cols, 0.0f), cols_(cols) {}
    Matrix3Xf(int rows, int cols) : data_(3 * cols, 0.0f), cols_(cols) {
        (void)rows; // rows is always 3
    }

    void resize(int rows, int cols) {
        (void)rows;
        cols_ = cols;
        data_.resize(3 * cols, 0.0f);
    }

    int rows() const { return 3; }
    int cols() const { return cols_; }

    float& operator()(int row, int col) { return data_[col * 3 + row]; }
    float operator()(int row, int col) const { return data_[col * 3 + row]; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    static Matrix3Xf Zero(int rows, int cols) {
        (void)rows;
        return Matrix3Xf(cols);
    }

    // Get column as Vector3f
    Vector3f col(int j) const {
        return Vector3f((*this)(0, j), (*this)(1, j), (*this)(2, j));
    }

    // Set column
    void setCol(int j, const Vector3f& v) {
        (*this)(0, j) = v(0);
        (*this)(1, j) = v(1);
        (*this)(2, j) = v(2);
    }

    // Row-wise operations (return a view or copy)
    class RowView {
    public:
        Matrix3Xf* mat;
        int row;

        RowView(Matrix3Xf* m, int r) : mat(m), row(r) {}

        float& operator()(int col) { return (*mat)(row, col); }
        float operator()(int col) const { return (*mat)(row, col); }

        // Mean of row
        float mean() const {
            float sum = 0.0f;
            for (int j = 0; j < mat->cols(); ++j) {
                sum += (*mat)(row, j);
            }
            return sum / mat->cols();
        }

        // cwiseProduct with another row
        RowView& operator*=(float s) {
            for (int j = 0; j < mat->cols(); ++j) {
                (*mat)(row, j) *= s;
            }
            return *this;
        }
    };

    RowView row(int i) { return RowView(this, i); }

    // Element-wise multiplication (broadcast vector to columns)
    Matrix3Xf cwiseProductColwise(const Vector3f& v) const {
        Matrix3Xf result(cols_);
        for (int j = 0; j < cols_; ++j) {
            result(0, j) = (*this)(0, j) * v(0);
            result(1, j) = (*this)(1, j) * v(1);
            result(2, j) = (*this)(2, j) * v(2);
        }
        return result;
    }

    // Row-wise multiplication with a row vector (VectorXf)
    Matrix3Xf rowwiseProduct(const VectorXf& rowVec) const {
        Matrix3Xf result(cols_);
#if MNN_USE_NEON
        NEON::mat3xN_rowwise_product_neon(data_.data(), rowVec.data(), result.data(), cols_);
#else
        for (int j = 0; j < cols_; ++j) {
            result(0, j) = (*this)(0, j) * rowVec(j);
            result(1, j) = (*this)(1, j) * rowVec(j);
            result(2, j) = (*this)(2, j) * rowVec(j);
        }
#endif
        return result;
    }

    // Add Vector3f to each column
    Matrix3Xf colwiseAdd(const Vector3f& v) const {
        Matrix3Xf result(cols_);
        for (int j = 0; j < cols_; ++j) {
            result(0, j) = (*this)(0, j) + v(0);
            result(1, j) = (*this)(1, j) + v(1);
            result(2, j) = (*this)(2, j) + v(2);
        }
        return result;
    }

    // Subtract another matrix
    Matrix3Xf operator-(const Matrix3Xf& other) const {
        Matrix3Xf result(cols_);
#if MNN_USE_NEON
        NEON::sub_neon(data_.data(), other.data_.data(), result.data(), 3 * cols_);
#else
        for (int i = 0; i < 3 * cols_; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
#endif
        return result;
    }

    // Add another matrix
    Matrix3Xf operator+(const Matrix3Xf& other) const {
        Matrix3Xf result(cols_);
#if MNN_USE_NEON
        NEON::add_neon(data_.data(), other.data_.data(), result.data(), 3 * cols_);
#else
        for (int i = 0; i < 3 * cols_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
#endif
        return result;
    }

    // Multiply 3x3 matrix from left: result = mat * this
    Matrix3Xf leftMultiply(const Matrix3f& mat) const {
        Matrix3Xf result(cols_);
        for (int j = 0; j < cols_; ++j) {
            for (int i = 0; i < 3; ++i) {
                result(i, j) = mat(i, 0) * (*this)(0, j) +
                               mat(i, 1) * (*this)(1, j) +
                               mat(i, 2) * (*this)(2, j);
            }
        }
        return result;
    }

    // Outer product: this * other^T -> 3x3 matrix
    Matrix3f outerProduct(const Matrix3Xf& other) const {
        Matrix3f result = Matrix3f::Zero();
#if MNN_USE_NEON
        NEON::mat3xN_outer_product_neon(data_.data(), other.data_.data(), result.data, cols_);
#else
        for (int j = 0; j < cols_; ++j) {
            for (int i = 0; i < 3; ++i) {
                for (int k = 0; k < 3; ++k) {
                    result(i, k) += (*this)(i, j) * other(k, j);
                }
            }
        }
#endif
        return result;
    }

    // Row-wise sum -> Vector3f
    Vector3f rowwiseSum() const {
        Vector3f result;
        for (int j = 0; j < cols_; ++j) {
            result(0) += (*this)(0, j);
            result(1) += (*this)(1, j);
            result(2) += (*this)(2, j);
        }
        return result;
    }

    // Element-wise product and sum (Frobenius inner product)
    float cwiseProductSum(const Matrix3Xf& other) const {
#if MNN_USE_NEON
        return NEON::dot_product_neon(data_.data(), other.data_.data(), 3 * cols_);
#else
        float sum = 0.0f;
        for (int i = 0; i < 3 * cols_; ++i) {
            sum += data_[i] * other.data_[i];
        }
        return sum;
#endif
    }

    // Change handedness (negate z-row)
    void negateRow(int r) {
        for (int j = 0; j < cols_; ++j) {
            (*this)(r, j) = -(*this)(r, j);
        }
    }

    // Copy from another matrix
    Matrix3Xf& operator=(const Matrix3Xf& other) {
        cols_ = other.cols_;
        data_ = other.data_;
        return *this;
    }
};


//=============================================================================
// SVD decomposition for 3x3 real matrix
// Uses one-sided Jacobi algorithm on A^T * A
// Returns U, singular values, V such that A = U * S * V^T
//=============================================================================
struct SVDResult3x3 {
    Matrix3f U;
    float singularValues[3];
    Matrix3f V;
};

// Compute Givens rotation to zero element (i, j) in symmetric matrix
// Returns (c, s) such that applying rotation zeros the (i, j) element
inline void computeSymmetricSchur2(const Matrix3f& A, int p, int q, float& c, float& s) {
    if (std::abs(A(p, q)) > std::numeric_limits<float>::epsilon()) {
        float tau = (A(q, q) - A(p, p)) / (2.0f * A(p, q));
        float t;
        if (tau >= 0.0f) {
            t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
        } else {
            t = 1.0f / (tau - std::sqrt(1.0f + tau * tau));
        }
        c = 1.0f / std::sqrt(1.0f + t * t);
        s = t * c;
    } else {
        c = 1.0f;
        s = 0.0f;
    }
}

// Apply Jacobi rotation to symmetric matrix: A = J^T * A * J
// This diagonalizes the 2x2 block at (p,p), (p,q), (q,p), (q,q)
inline void jacobiRotateSymmetric(Matrix3f& A, int p, int q, float c, float s) {
    float App = A(p, p);
    float Aqq = A(q, q);
    float Apq = A(p, q);

    A(p, p) = c * c * App - 2.0f * c * s * Apq + s * s * Aqq;
    A(q, q) = s * s * App + 2.0f * c * s * Apq + c * c * Aqq;
    A(p, q) = A(q, p) = 0.0f;  // This is the goal of Jacobi rotation

    // Update other elements
    for (int i = 0; i < 3; ++i) {
        if (i != p && i != q) {
            float Aip = A(i, p);
            float Aiq = A(i, q);
            A(i, p) = A(p, i) = c * Aip - s * Aiq;
            A(i, q) = A(q, i) = s * Aip + c * Aiq;
        }
    }
}

// Apply Jacobi rotation to matrix on the right: A = A * J
inline void jacobiRotateRight(Matrix3f& A, int p, int q, float c, float s) {
#if MNN_USE_NEON
    NEON::jacobi_rotate_right_neon(A.data, p, q, c, s);
#else
    for (int i = 0; i < 3; ++i) {
        float Aip = A(i, p);
        float Aiq = A(i, q);
        A(i, p) = c * Aip - s * Aiq;
        A(i, q) = s * Aip + c * Aiq;
    }
#endif
}

inline SVDResult3x3 computeSVD3x3(const Matrix3f& input) {
    SVDResult3x3 result;

    // Compute A^T * A (symmetric)
    Matrix3f AtA = input.transpose() * input;

    // V will accumulate the eigenvectors of A^T * A
    result.V = Matrix3f::Identity();

    // Jacobi iteration to diagonalize A^T * A
    const int maxIterations = 50;
    const float tolerance = 1e-10f;

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Find largest off-diagonal element
        float maxOffDiag = 0.0f;
        int p = 0, q = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                if (std::abs(AtA(i, j)) > maxOffDiag) {
                    maxOffDiag = std::abs(AtA(i, j));
                    p = i;
                    q = j;
                }
            }
        }

        if (maxOffDiag < tolerance) break;

        // Compute Jacobi rotation
        float c, s;
        computeSymmetricSchur2(AtA, p, q, c, s);

        // Apply rotation to AtA: AtA = J^T * AtA * J
        jacobiRotateSymmetric(AtA, p, q, c, s);

        // Accumulate V: V = V * J
        jacobiRotateRight(result.V, p, q, c, s);
    }

    // Extract singular values (sqrt of eigenvalues of A^T * A)
    // and sort in descending order
    float eigenvalues[3] = {AtA(0, 0), AtA(1, 1), AtA(2, 2)};
    int order[3] = {0, 1, 2};

    // Selection sort for 3 elements - sort descending by eigenvalue
    for (int i = 0; i < 2; ++i) {
        int maxIdx = i;
        for (int j = i + 1; j < 3; ++j) {
            if (eigenvalues[order[j]] > eigenvalues[order[maxIdx]]) {
                maxIdx = j;
            }
        }
        if (maxIdx != i) {
            std::swap(order[i], order[maxIdx]);
        }
    }

    // Reorder V and compute singular values
    Matrix3f V_sorted;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            V_sorted(j, i) = result.V(j, order[i]);
        }
        float ev = eigenvalues[order[i]];
        result.singularValues[i] = (ev > 0.0f) ? std::sqrt(ev) : 0.0f;
    }
    result.V = V_sorted;

    // Compute U = A * V * S^(-1)
    // U(:, i) = A * V(:, i) / sigma_i
    result.U = Matrix3f::Zero();
    int numValidSingularValues = 0;
    for (int i = 0; i < 3; ++i) {
        if (result.singularValues[i] > 1e-10f) {
            Vector3f vi = result.V.col(i);
            Vector3f Avi = input * vi;
            float invSigma = 1.0f / result.singularValues[i];
            result.U.setCol(i, Avi * invSigma);
            numValidSingularValues++;
        }
    }

    // For rank-deficient matrices, complete U to orthonormal basis
    if (numValidSingularValues < 3) {
        // Use cross products to find orthogonal vectors
        for (int i = numValidSingularValues; i < 3; ++i) {
            Vector3f ui;
            if (i == 0) {
                ui = Vector3f(1, 0, 0);
            } else if (i == 1) {
                Vector3f u0 = result.U.col(0);
                Vector3f candidate(1, 0, 0);
                if (std::abs(u0(0)) > 0.9f) candidate = Vector3f(0, 1, 0);
                // Cross product
                ui = Vector3f(
                    u0(1) * candidate(2) - u0(2) * candidate(1),
                    u0(2) * candidate(0) - u0(0) * candidate(2),
                    u0(0) * candidate(1) - u0(1) * candidate(0)
                );
                float norm = ui.norm();
                if (norm > 1e-10f) ui = ui * (1.0f / norm);
            } else { // i == 2
                Vector3f u0 = result.U.col(0);
                Vector3f u1 = result.U.col(1);
                ui = Vector3f(
                    u0(1) * u1(2) - u0(2) * u1(1),
                    u0(2) * u1(0) - u0(0) * u1(2),
                    u0(0) * u1(1) - u0(1) * u1(0)
                );
                float norm = ui.norm();
                if (norm > 1e-10f) ui = ui * (1.0f / norm);
            }
            result.U.setCol(i, ui);
        }
    }


    return result;
}

//=============================================================================
// Homogeneous coordinate helpers
//=============================================================================

// Apply 4x4 transform to 3xN points (homogeneous: w=1)
// result = (M * [points; 1]).topRows(3)
inline Matrix3Xf applyTransform(const Matrix4f& M, const Matrix3Xf& points) {
    Matrix3Xf result(points.cols());
#if MNN_USE_NEON
    NEON::apply_transform_4x4_neon(M.data, points.data(), result.data(), points.cols());
#else
    for (int j = 0; j < points.cols(); ++j) {
        float x = points(0, j);
        float y = points(1, j);
        float z = points(2, j);
        // Homogeneous: w = 1
        result(0, j) = M(0, 0) * x + M(0, 1) * y + M(0, 2) * z + M(0, 3);
        result(1, j) = M(1, 0) * x + M(1, 1) * y + M(1, 2) * z + M(1, 3);
        result(2, j) = M(2, 0) * x + M(2, 1) * y + M(2, 2) * z + M(2, 3);
    }
#endif
    return result;
}

} // namespace MNN::FaceGeometry


#endif  // MNN_FACE_GEOMETRY_MINI_LINALG_H

