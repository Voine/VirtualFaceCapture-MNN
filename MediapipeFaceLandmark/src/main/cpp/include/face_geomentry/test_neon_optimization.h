/**
 * test_neon_optimization.h
 *
 * Test and benchmark for NEON optimization in Face Geometry.
 * Include this file and call runNeonTests() to verify correctness
 * and measure performance gains.
 *
 * Usage (Android JNI):
 *   Call NeonOptimizationTest.runTests() from Java/Kotlin
 *
 * Usage (Desktop):
 *   #include "test_neon_optimization.h"
 *   MNN::FaceGeometry::runNeonTests();
 */

#ifndef MNN_FACE_GEOMETRY_TEST_NEON_OPTIMIZATION_H
#define MNN_FACE_GEOMETRY_TEST_NEON_OPTIMIZATION_H

#include "mini_linalg.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Android logging support
#ifdef __ANDROID__
#include <android/log.h>
#define NEON_TEST_TAG "NeonOptTest"
#define NEON_LOG_I(...) __android_log_print(ANDROID_LOG_INFO, NEON_TEST_TAG, __VA_ARGS__)
#define NEON_LOG_D(...) __android_log_print(ANDROID_LOG_DEBUG, NEON_TEST_TAG, __VA_ARGS__)
#define NEON_LOG_E(...) __android_log_print(ANDROID_LOG_ERROR, NEON_TEST_TAG, __VA_ARGS__)
#else
#define NEON_LOG_I(...) printf(__VA_ARGS__); printf("\n")
#define NEON_LOG_D(...) printf(__VA_ARGS__); printf("\n")
#define NEON_LOG_E(...) printf(__VA_ARGS__); printf("\n")
#endif

namespace MNN {
namespace FaceGeometry {

//=============================================================================
// Test utilities
//=============================================================================

class NeonTestRunner {
public:
    static constexpr int NUM_LANDMARKS = 468;
    static constexpr int NUM_ITERATIONS = 1000;
    static constexpr float TOLERANCE = 1e-4f;

    static void printHeader(const char* title) {
        NEON_LOG_I("========================================");
        NEON_LOG_I(" %s", title);
        NEON_LOG_I("========================================");
    }

    static void printResult(const char* name, bool passed, double time_ms = -1) {
        if (time_ms >= 0) {
            if (passed) {
                NEON_LOG_I("  [PASS] %s (%.3f ms / %d iters)", name, time_ms, NUM_ITERATIONS);
            } else {
                NEON_LOG_E("  [FAIL] %s (%.3f ms / %d iters)", name, time_ms, NUM_ITERATIONS);
            }
        } else {
            if (passed) {
                NEON_LOG_I("  [PASS] %s", name);
            } else {
                NEON_LOG_E("  [FAIL] %s", name);
            }
        }
    }

    static float randomFloat() {
        return static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    static bool floatEquals(float a, float b, float tol = TOLERANCE) {
        return std::abs(a - b) < tol;
    }

    static bool vectorEquals(const Vector3f& a, const Vector3f& b, float tol = TOLERANCE) {
        return floatEquals(a(0), b(0), tol) &&
               floatEquals(a(1), b(1), tol) &&
               floatEquals(a(2), b(2), tol);
    }

    static bool matrix3fEquals(const Matrix3f& a, const Matrix3f& b, float tol = TOLERANCE) {
        for (int i = 0; i < 9; ++i) {
            if (!floatEquals(a.data[i], b.data[i], tol)) return false;
        }
        return true;
    }

    static bool matrix3XfEquals(const Matrix3Xf& a, const Matrix3Xf& b, float tol = TOLERANCE) {
        if (a.cols() != b.cols()) return false;
        for (int j = 0; j < a.cols(); ++j) {
            for (int i = 0; i < 3; ++i) {
                if (!floatEquals(a(i, j), b(i, j), tol)) return false;
            }
        }
        return true;
    }

    //=========================================================================
    // Individual tests
    //=========================================================================

    static bool testVectorXfSum() {
        VectorXf v(NUM_LANDMARKS);
        for (int i = 0; i < v.size(); ++i) {
            v(i) = randomFloat();
        }

        // Reference implementation
        float expected = 0.0f;
        for (int i = 0; i < v.size(); ++i) {
            expected += v(i);
        }

        float actual = v.sum();
        return floatEquals(expected, actual, 1e-2f);  // Larger tolerance for sum
    }

    static bool testVectorXfCwiseProduct() {
        VectorXf a(NUM_LANDMARKS), b(NUM_LANDMARKS);
        for (int i = 0; i < NUM_LANDMARKS; ++i) {
            a(i) = randomFloat();
            b(i) = randomFloat();
        }

        VectorXf result = a.cwiseProduct(b);

        for (int i = 0; i < NUM_LANDMARKS; ++i) {
            if (!floatEquals(result(i), a(i) * b(i))) return false;
        }
        return true;
    }

    static bool testMatrix3fMultiply() {
        Matrix3f a, b;
        for (int i = 0; i < 9; ++i) {
            a.data[i] = randomFloat();
            b.data[i] = randomFloat();
        }

        // Reference implementation
        Matrix3f expected;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                expected(i, j) = sum;
            }
        }

        Matrix3f actual = a * b;
        return matrix3fEquals(expected, actual);
    }

    static bool testMatrix3XfCwiseProductSum() {
        Matrix3Xf a(NUM_LANDMARKS), b(NUM_LANDMARKS);
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                a(i, j) = randomFloat();
                b(i, j) = randomFloat();
            }
        }

        // Reference implementation
        float expected = 0.0f;
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                expected += a(i, j) * b(i, j);
            }
        }

        float actual = a.cwiseProductSum(b);
        return floatEquals(expected, actual, 1e-1f);  // Larger tolerance
    }

    static bool testMatrix3XfRowwiseProduct() {
        Matrix3Xf a(NUM_LANDMARKS);
        VectorXf w(NUM_LANDMARKS);
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            w(j) = randomFloat();
            for (int i = 0; i < 3; ++i) {
                a(i, j) = randomFloat();
            }
        }

        Matrix3Xf result = a.rowwiseProduct(w);

        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                if (!floatEquals(result(i, j), a(i, j) * w(j))) return false;
            }
        }
        return true;
    }

    static bool testMatrix3XfLeftMultiply() {
        Matrix3f mat;
        Matrix3Xf points(NUM_LANDMARKS);
        for (int i = 0; i < 9; ++i) mat.data[i] = randomFloat();
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                points(i, j) = randomFloat();
            }
        }

        Matrix3Xf result = points.leftMultiply(mat);

        // Verify first few points
        for (int j = 0; j < std::min(10, NUM_LANDMARKS); ++j) {
            for (int i = 0; i < 3; ++i) {
                float expected = mat(i, 0) * points(0, j) +
                                 mat(i, 1) * points(1, j) +
                                 mat(i, 2) * points(2, j);
                if (!floatEquals(result(i, j), expected)) return false;
            }
        }
        return true;
    }

    static bool testMatrix3XfOuterProduct() {
        Matrix3Xf a(NUM_LANDMARKS), b(NUM_LANDMARKS);
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                a(i, j) = randomFloat() * 0.1f;  // Scale down to avoid overflow
                b(i, j) = randomFloat() * 0.1f;
            }
        }

        // Reference implementation
        Matrix3f expected = Matrix3f::Zero();
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                for (int k = 0; k < 3; ++k) {
                    expected(i, k) += a(i, j) * b(k, j);
                }
            }
        }

        Matrix3f actual = a.outerProduct(b);
        return matrix3fEquals(expected, actual, 1e-2f);
    }

    static bool testMatrix3XfRowwiseSum() {
        Matrix3Xf a(NUM_LANDMARKS);
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                a(i, j) = randomFloat();
            }
        }

        // Reference implementation
        Vector3f expected;
        expected(0) = expected(1) = expected(2) = 0.0f;
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            expected(0) += a(0, j);
            expected(1) += a(1, j);
            expected(2) += a(2, j);
        }

        Vector3f actual = a.rowwiseSum();
        return vectorEquals(expected, actual, 1e-1f);
    }

    static bool testApplyTransform() {
        Matrix4f M = Matrix4f::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                M(i, j) = randomFloat();
            }
            M(i, 3) = randomFloat() * 10.0f;  // Translation
        }

        Matrix3Xf points(NUM_LANDMARKS);
        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            for (int i = 0; i < 3; ++i) {
                points(i, j) = randomFloat() * 100.0f;
            }
        }

        Matrix3Xf result = applyTransform(M, points);

        // Verify first few points
        for (int j = 0; j < std::min(10, NUM_LANDMARKS); ++j) {
            float x = points(0, j), y = points(1, j), z = points(2, j);
            float ex = M(0, 0) * x + M(0, 1) * y + M(0, 2) * z + M(0, 3);
            float ey = M(1, 0) * x + M(1, 1) * y + M(1, 2) * z + M(1, 3);
            float ez = M(2, 0) * x + M(2, 1) * y + M(2, 2) * z + M(2, 3);

            if (!floatEquals(result(0, j), ex) ||
                !floatEquals(result(1, j), ey) ||
                !floatEquals(result(2, j), ez)) {
                return false;
            }
        }
        return true;
    }

    static bool testSVD3x3() {
        Matrix3f A;
        for (int i = 0; i < 9; ++i) {
            A.data[i] = randomFloat();
        }

        SVDResult3x3 svd = computeSVD3x3(A);

        // Verify: A ≈ U * S * V^T
        Matrix3f S = Matrix3f::Zero();
        S(0, 0) = svd.singularValues[0];
        S(1, 1) = svd.singularValues[1];
        S(2, 2) = svd.singularValues[2];

        Matrix3f reconstructed = svd.U * S * svd.V.transpose();
        return matrix3fEquals(A, reconstructed, 1e-3f);
    }

    //=========================================================================
    // Benchmark
    //=========================================================================

    template<typename Func>
    static double benchmark(Func f) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            f();
        }
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static void runBenchmarks() {
        printHeader("Performance Benchmarks");

        // Prepare test data
        Matrix3Xf points(NUM_LANDMARKS);
        Matrix3Xf points2(NUM_LANDMARKS);
        VectorXf weights(NUM_LANDMARKS);
        Matrix3f mat3;
        Matrix4f mat4 = Matrix4f::Identity();

        for (int j = 0; j < NUM_LANDMARKS; ++j) {
            weights(j) = randomFloat();
            for (int i = 0; i < 3; ++i) {
                points(i, j) = randomFloat();
                points2(i, j) = randomFloat();
            }
        }
        for (int i = 0; i < 9; ++i) mat3.data[i] = randomFloat();
        for (int i = 0; i < 16; ++i) mat4.data[i] = randomFloat();

        // Run benchmarks
        // Use sink variables to prevent compiler from optimizing away the code
        float sink_f = 0.0f;
        Matrix3Xf sink_m3x(NUM_LANDMARKS);
        Matrix3f sink_m3;
        Vector3f sink_v3;

        double t1 = benchmark([&]() { sink_f += points.cwiseProductSum(points2); });
        NEON_LOG_I("  cwiseProductSum: %.3f ms / %d iters", t1, NUM_ITERATIONS);

        double t2 = benchmark([&]() { sink_m3x = points.rowwiseProduct(weights); });
        NEON_LOG_I("  rowwiseProduct:  %.3f ms / %d iters", t2, NUM_ITERATIONS);

        double t3 = benchmark([&]() { sink_m3x = points.leftMultiply(mat3); });
        NEON_LOG_I("  leftMultiply:    %.3f ms / %d iters", t3, NUM_ITERATIONS);

        double t4 = benchmark([&]() { sink_m3 = points.outerProduct(points2); });
        NEON_LOG_I("  outerProduct:    %.3f ms / %d iters", t4, NUM_ITERATIONS);

        double t5 = benchmark([&]() { sink_v3 = points.rowwiseSum(); });
        NEON_LOG_I("  rowwiseSum:      %.3f ms / %d iters", t5, NUM_ITERATIONS);

        double t6 = benchmark([&]() { sink_m3x = applyTransform(mat4, points); });
        NEON_LOG_I("  applyTransform:  %.3f ms / %d iters", t6, NUM_ITERATIONS);

        double t7 = benchmark([&]() { sink_m3x = points - points2; });
        NEON_LOG_I("  Matrix3Xf sub:   %.3f ms / %d iters", t7, NUM_ITERATIONS);

        double t8 = benchmark([&]() {
            SVDResult3x3 svd = computeSVD3x3(mat3);
            (void)svd;
        });
        NEON_LOG_I("  SVD 3x3:         %.3f ms / %d iters", t8, NUM_ITERATIONS);

        // Prevent optimization by using the sink values
        if (sink_f == 0.0f && sink_m3x.cols() == 0 && sink_m3.data[0] == 0.0f && sink_v3(0) == 0.0f) {
            NEON_LOG_D("  (sink values used to prevent optimization)");
        }
    }
};

//=============================================================================
// Main test runner
//=============================================================================

inline void runNeonTests() {
    srand(42);  // Fixed seed for reproducibility

    NeonTestRunner::printHeader("NEON Optimization Tests");

#if MNN_USE_NEON
    NEON_LOG_I("  NEON: ENABLED");
#if MNN_USE_NEON64
    NEON_LOG_I("  Mode: ARM64 (native vaddvq_f32)");
#else
    NEON_LOG_I("  Mode: ARM32 (compat vaddvq_f32)");
#endif
#else
    NEON_LOG_I("  NEON: DISABLED (scalar fallback)");
#endif

    // Run correctness tests
    NeonTestRunner::printHeader("Correctness Tests");

    NeonTestRunner::printResult("VectorXf::sum()",
        NeonTestRunner::testVectorXfSum());

    NeonTestRunner::printResult("VectorXf::cwiseProduct()",
        NeonTestRunner::testVectorXfCwiseProduct());

    NeonTestRunner::printResult("Matrix3f::operator*()",
        NeonTestRunner::testMatrix3fMultiply());

    NeonTestRunner::printResult("Matrix3Xf::cwiseProductSum()",
        NeonTestRunner::testMatrix3XfCwiseProductSum());

    NeonTestRunner::printResult("Matrix3Xf::rowwiseProduct()",
        NeonTestRunner::testMatrix3XfRowwiseProduct());

    NeonTestRunner::printResult("Matrix3Xf::leftMultiply()",
        NeonTestRunner::testMatrix3XfLeftMultiply());

    NeonTestRunner::printResult("Matrix3Xf::outerProduct()",
        NeonTestRunner::testMatrix3XfOuterProduct());

    NeonTestRunner::printResult("Matrix3Xf::rowwiseSum()",
        NeonTestRunner::testMatrix3XfRowwiseSum());

    NeonTestRunner::printResult("applyTransform()",
        NeonTestRunner::testApplyTransform());

    NeonTestRunner::printResult("computeSVD3x3()",
        NeonTestRunner::testSVD3x3());

    // Run benchmarks
    NeonTestRunner::runBenchmarks();

    NEON_LOG_I("NEON Optimization Tests Complete!");
}

}  // namespace FaceGeometry
}  // namespace MNN

#endif  // MNN_FACE_GEOMETRY_TEST_NEON_OPTIMIZATION_H
