/**
 * neon_intrinsics.h
 *
 * NEON intrinsics wrapper and optimized functions for Face Geometry.
 * Provides SIMD acceleration for ARM processors (Android/iOS).
 *
 */

#ifndef MNN_FACE_GEOMETRY_NEON_INTRINSICS_H
#define MNN_FACE_GEOMETRY_NEON_INTRINSICS_H

// Detect NEON support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define MNN_USE_NEON 1
    #include <arm_neon.h>
#else
    #define MNN_USE_NEON 0
#endif

// For ARM64, we have additional instructions
#if defined(__aarch64__) || defined(_M_ARM64)
    #define MNN_USE_NEON64 1
#else
    #define MNN_USE_NEON64 0
#endif



namespace MNN::FaceGeometry::NEON {

#if MNN_USE_NEON

//=============================================================================
// Horizontal sum for float32x4_t
//=============================================================================
inline float vaddvq_f32_compat(float32x4_t v) {
#if MNN_USE_NEON64
    // ARM64 has native vaddvq_f32
    return vaddvq_f32(v);
#else
    // ARM32 fallback: pairwise add
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}

//=============================================================================
// Dot product of two float arrays
//=============================================================================
inline float dot_product_neon(const float* a, const float* b, int n) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;

    // Process 16 elements at a time (4 x 4-way SIMD)
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vsum = vmlaq_f32(vsum, a0, b0);
        vsum = vmlaq_f32(vsum, a1, b1);
        vsum = vmlaq_f32(vsum, a2, b2);
        vsum = vmlaq_f32(vsum, a3, b3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    float result = vaddvq_f32_compat(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

//=============================================================================
// Sum of float array
//=============================================================================
inline float sum_neon(const float* a, int n) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;

    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        vsum = vaddq_f32(vsum, a0);
        vsum = vaddq_f32(vsum, a1);
        vsum = vaddq_f32(vsum, a2);
        vsum = vaddq_f32(vsum, a3);
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vsum = vaddq_f32(vsum, va);
    }

    float result = vaddvq_f32_compat(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        result += a[i];
    }

    return result;
}

//=============================================================================
// Element-wise vector operations: c = a * b
//=============================================================================
inline void mul_neon(const float* a, const float* b, float* c, int n) {
    int i = 0;

    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(c + i, vmulq_f32(a0, b0));
        vst1q_f32(c + i + 4, vmulq_f32(a1, b1));
        vst1q_f32(c + i + 8, vmulq_f32(a2, b2));
        vst1q_f32(c + i + 12, vmulq_f32(a3, b3));
    }

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(c + i, vmulq_f32(va, vb));
    }

    // Scalar tail
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

//=============================================================================
// Element-wise: c = a + b
//=============================================================================
inline void add_neon(const float* a, const float* b, float* c, int n) {
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        vst1q_f32(c + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(c + i + 4, vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
        vst1q_f32(c + i + 8, vaddq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8)));
        vst1q_f32(c + i + 12, vaddq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)));
    }

    for (; i + 4 <= n; i += 4) {
        vst1q_f32(c + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

//=============================================================================
// Element-wise: c = a - b
//=============================================================================
inline void sub_neon(const float* a, const float* b, float* c, int n) {
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        vst1q_f32(c + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(c + i + 4, vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
        vst1q_f32(c + i + 8, vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8)));
        vst1q_f32(c + i + 12, vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)));
    }

    for (; i + 4 <= n; i += 4) {
        vst1q_f32(c + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }

    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

//=============================================================================
// Broadcast scalar multiply: c = a * scalar
//=============================================================================
inline void scale_neon(const float* a, float scalar, float* c, int n) {
    float32x4_t vs = vdupq_n_f32(scalar);
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        vst1q_f32(c + i, vmulq_f32(vld1q_f32(a + i), vs));
        vst1q_f32(c + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vs));
        vst1q_f32(c + i + 8, vmulq_f32(vld1q_f32(a + i + 8), vs));
        vst1q_f32(c + i + 12, vmulq_f32(vld1q_f32(a + i + 12), vs));
    }

    for (; i + 4 <= n; i += 4) {
        vst1q_f32(c + i, vmulq_f32(vld1q_f32(a + i), vs));
    }

    for (; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
}

//=============================================================================
// 3x3 Matrix multiplication: C = A * B
//=============================================================================
inline void mat3_mul_neon(const float* A, const float* B, float* C) {
    // A and B are column-major: A[col*3 + row]
    // Load columns of B
    float32x4_t b0 = {B[0], B[1], B[2], 0};  // col 0
    float32x4_t b1 = {B[3], B[4], B[5], 0};  // col 1
    float32x4_t b2 = {B[6], B[7], B[8], 0};  // col 2

    // For each column of result
    for (int j = 0; j < 3; ++j) {
        float32x4_t aj = {A[j*3], A[j*3+1], A[j*3+2], 0};
        
        // C(:,j) = A(:,0)*B(0,j) + A(:,1)*B(1,j) + A(:,2)*B(2,j)
        float32x4_t a0 = {A[0], A[1], A[2], 0};
        float32x4_t a1 = {A[3], A[4], A[5], 0};
        float32x4_t a2 = {A[6], A[7], A[8], 0};

        float32x4_t cj = vmulq_n_f32(a0, B[j*3]);
        cj = vmlaq_n_f32(cj, a1, B[j*3+1]);
        cj = vmlaq_n_f32(cj, a2, B[j*3+2]);

        C[j*3]   = vgetq_lane_f32(cj, 0);
        C[j*3+1] = vgetq_lane_f32(cj, 1);
        C[j*3+2] = vgetq_lane_f32(cj, 2);
    }
}

//=============================================================================
// 3xN matrix left multiply by 3x3: result = mat * points
// points is column-major 3xN, mat is column-major 3x3
//=============================================================================
inline void mat3xN_left_multiply_neon(
    const float* mat,      // 3x3 column-major
    const float* points,   // 3xN column-major
    float* result,         // 3xN column-major output
    int n_cols)
{
    // Load matrix rows (transposed access for row-major multiplication)
    // mat is column-major, so mat(i,j) = mat[j*3 + i]
    // Row 0: mat(0,0), mat(0,1), mat(0,2)
    float m00 = mat[0], m01 = mat[3], m02 = mat[6];
    float m10 = mat[1], m11 = mat[4], m12 = mat[7];
    float m20 = mat[2], m21 = mat[5], m22 = mat[8];

    float32x4_t vm00 = vdupq_n_f32(m00);
    float32x4_t vm01 = vdupq_n_f32(m01);
    float32x4_t vm02 = vdupq_n_f32(m02);
    float32x4_t vm10 = vdupq_n_f32(m10);
    float32x4_t vm11 = vdupq_n_f32(m11);
    float32x4_t vm12 = vdupq_n_f32(m12);
    float32x4_t vm20 = vdupq_n_f32(m20);
    float32x4_t vm21 = vdupq_n_f32(m21);
    float32x4_t vm22 = vdupq_n_f32(m22);

    int j = 0;

    // Process 4 columns at a time with deinterleaving
    // Column-major: data layout is [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3, ...]
    for (; j + 4 <= n_cols; j += 4) {
        // Load 4 columns (12 floats)
        float32x4x3_t cols = vld3q_f32(points + j * 3);
        float32x4_t vx = cols.val[0];  // x0, x1, x2, x3
        float32x4_t vy = cols.val[1];  // y0, y1, y2, y3
        float32x4_t vz = cols.val[2];  // z0, z1, z2, z3

        // Compute result rows
        float32x4_t rx = vmlaq_f32(vmlaq_f32(vmulq_f32(vm00, vx), vm01, vy), vm02, vz);
        float32x4_t ry = vmlaq_f32(vmlaq_f32(vmulq_f32(vm10, vx), vm11, vy), vm12, vz);
        float32x4_t rz = vmlaq_f32(vmlaq_f32(vmulq_f32(vm20, vx), vm21, vy), vm22, vz);

        // Store interleaved
        float32x4x3_t res = {rx, ry, rz};
        vst3q_f32(result + j * 3, res);
    }

    // Scalar tail
    for (; j < n_cols; ++j) {
        float x = points[j * 3];
        float y = points[j * 3 + 1];
        float z = points[j * 3 + 2];
        result[j * 3]     = m00 * x + m01 * y + m02 * z;
        result[j * 3 + 1] = m10 * x + m11 * y + m12 * z;
        result[j * 3 + 2] = m20 * x + m21 * y + m22 * z;
    }
}

//=============================================================================
// Apply 4x4 homogeneous transform to 3xN points
// M is 4x4 column-major, points is 3xN column-major
//=============================================================================
inline void apply_transform_4x4_neon(
    const float* M,        // 4x4 column-major
    const float* points,   // 3xN column-major
    float* result,         // 3xN column-major output
    int n_cols)
{
    // Extract 3x3 rotation and translation from M
    // M is column-major: M(i,j) = M[j*4 + i]
    float m00 = M[0], m01 = M[4], m02 = M[8],  m03 = M[12];
    float m10 = M[1], m11 = M[5], m12 = M[9],  m13 = M[13];
    float m20 = M[2], m21 = M[6], m22 = M[10], m23 = M[14];

    float32x4_t vm00 = vdupq_n_f32(m00);
    float32x4_t vm01 = vdupq_n_f32(m01);
    float32x4_t vm02 = vdupq_n_f32(m02);
    float32x4_t vm03 = vdupq_n_f32(m03);
    float32x4_t vm10 = vdupq_n_f32(m10);
    float32x4_t vm11 = vdupq_n_f32(m11);
    float32x4_t vm12 = vdupq_n_f32(m12);
    float32x4_t vm13 = vdupq_n_f32(m13);
    float32x4_t vm20 = vdupq_n_f32(m20);
    float32x4_t vm21 = vdupq_n_f32(m21);
    float32x4_t vm22 = vdupq_n_f32(m22);
    float32x4_t vm23 = vdupq_n_f32(m23);

    int j = 0;

    // Process 4 columns at a time
    for (; j + 4 <= n_cols; j += 4) {
        float32x4x3_t cols = vld3q_f32(points + j * 3);
        float32x4_t vx = cols.val[0];
        float32x4_t vy = cols.val[1];
        float32x4_t vz = cols.val[2];

        // result = M * [x; y; z; 1]
        float32x4_t rx = vaddq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(vm00, vx), vm01, vy), vm02, vz), vm03);
        float32x4_t ry = vaddq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(vm10, vx), vm11, vy), vm12, vz), vm13);
        float32x4_t rz = vaddq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(vm20, vx), vm21, vy), vm22, vz), vm23);

        float32x4x3_t res = {rx, ry, rz};
        vst3q_f32(result + j * 3, res);
    }

    // Scalar tail
    for (; j < n_cols; ++j) {
        float x = points[j * 3];
        float y = points[j * 3 + 1];
        float z = points[j * 3 + 2];
        result[j * 3]     = m00 * x + m01 * y + m02 * z + m03;
        result[j * 3 + 1] = m10 * x + m11 * y + m12 * z + m13;
        result[j * 3 + 2] = m20 * x + m21 * y + m22 * z + m23;
    }
}

//=============================================================================
// 3xN row-wise product with 1xN vector (broadcast each weight to 3 elements)
//=============================================================================
inline void mat3xN_rowwise_product_neon(
    const float* mat,      // 3xN column-major
    const float* weights,  // N weights
    float* result,         // 3xN column-major output
    int n_cols)
{
    int j = 0;

    // Process 4 columns at a time
    for (; j + 4 <= n_cols; j += 4) {
        float32x4_t w = vld1q_f32(weights + j);

        // Load 4 columns
        float32x4x3_t cols = vld3q_f32(mat + j * 3);

        // Multiply each component by weight
        cols.val[0] = vmulq_f32(cols.val[0], w);
        cols.val[1] = vmulq_f32(cols.val[1], w);
        cols.val[2] = vmulq_f32(cols.val[2], w);

        vst3q_f32(result + j * 3, cols);
    }

    // Scalar tail
    for (; j < n_cols; ++j) {
        float w = weights[j];
        result[j * 3]     = mat[j * 3] * w;
        result[j * 3 + 1] = mat[j * 3 + 1] * w;
        result[j * 3 + 2] = mat[j * 3 + 2] * w;
    }
}

//=============================================================================
// 3xN row-wise sum -> Vector3f
//=============================================================================
inline void mat3xN_rowwise_sum_neon(
    const float* mat,  // 3xN column-major
    float* result,     // 3-element output
    int n_cols)
{
    float32x4_t sum_x = vdupq_n_f32(0.0f);
    float32x4_t sum_y = vdupq_n_f32(0.0f);
    float32x4_t sum_z = vdupq_n_f32(0.0f);

    int j = 0;

    // Process 4 columns at a time
    for (; j + 4 <= n_cols; j += 4) {
        float32x4x3_t cols = vld3q_f32(mat + j * 3);
        sum_x = vaddq_f32(sum_x, cols.val[0]);
        sum_y = vaddq_f32(sum_y, cols.val[1]);
        sum_z = vaddq_f32(sum_z, cols.val[2]);
    }

    result[0] = vaddvq_f32_compat(sum_x);
    result[1] = vaddvq_f32_compat(sum_y);
    result[2] = vaddvq_f32_compat(sum_z);

    // Scalar tail
    for (; j < n_cols; ++j) {
        result[0] += mat[j * 3];
        result[1] += mat[j * 3 + 1];
        result[2] += mat[j * 3 + 2];
    }
}

//=============================================================================
// Outer product for Procrustes: A * B^T where A and B are 3xN
// Result is 3x3 matrix
//=============================================================================
inline void mat3xN_outer_product_neon(
    const float* A,    // 3xN column-major
    const float* B,    // 3xN column-major
    float* C,          // 3x3 column-major output
    int n_cols)
{
    // C(i, k) = sum_j A(i, j) * B(k, j)
    // Initialize result to zero
    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c02 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c12 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c22 = vdupq_n_f32(0.0f);

    int j = 0;

    // Process 4 columns at a time
    for (; j + 4 <= n_cols; j += 4) {
        float32x4x3_t colsA = vld3q_f32(A + j * 3);
        float32x4x3_t colsB = vld3q_f32(B + j * 3);

        float32x4_t ax = colsA.val[0];
        float32x4_t ay = colsA.val[1];
        float32x4_t az = colsA.val[2];
        float32x4_t bx = colsB.val[0];
        float32x4_t by = colsB.val[1];
        float32x4_t bz = colsB.val[2];

        // Accumulate outer products
        c00 = vmlaq_f32(c00, ax, bx);
        c01 = vmlaq_f32(c01, ax, by);
        c02 = vmlaq_f32(c02, ax, bz);
        c10 = vmlaq_f32(c10, ay, bx);
        c11 = vmlaq_f32(c11, ay, by);
        c12 = vmlaq_f32(c12, ay, bz);
        c20 = vmlaq_f32(c20, az, bx);
        c21 = vmlaq_f32(c21, az, by);
        c22 = vmlaq_f32(c22, az, bz);
    }

    // Horizontal sum all accumulators
    C[0] = vaddvq_f32_compat(c00);
    C[1] = vaddvq_f32_compat(c10);
    C[2] = vaddvq_f32_compat(c20);
    C[3] = vaddvq_f32_compat(c01);
    C[4] = vaddvq_f32_compat(c11);
    C[5] = vaddvq_f32_compat(c21);
    C[6] = vaddvq_f32_compat(c02);
    C[7] = vaddvq_f32_compat(c12);
    C[8] = vaddvq_f32_compat(c22);

    // Scalar tail
    for (; j < n_cols; ++j) {
        float ax = A[j * 3];
        float ay = A[j * 3 + 1];
        float az = A[j * 3 + 2];
        float bx = B[j * 3];
        float by = B[j * 3 + 1];
        float bz = B[j * 3 + 2];

        C[0] += ax * bx;
        C[1] += ay * bx;
        C[2] += az * bx;
        C[3] += ax * by;
        C[4] += ay * by;
        C[5] += az * by;
        C[6] += ax * bz;
        C[7] += ay * bz;
        C[8] += az * bz;
    }
}

//=============================================================================
// Jacobi rotation helpers for SVD
//=============================================================================
inline void jacobi_rotate_right_neon(float* A, int p, int q, float c, float s) {
    // A is 3x3 column-major
    // Apply rotation on right: A = A * J
    // where J is rotation in plane (p, q)
    float32x4_t vc = vdupq_n_f32(c);
    float32x4_t vs = vdupq_n_f32(s);

    // Load columns p and q
    float32x4_t colp = {A[p*3], A[p*3+1], A[p*3+2], 0};
    float32x4_t colq = {A[q*3], A[q*3+1], A[q*3+2], 0};

    // new_p = c * colp - s * colq
    // new_q = s * colp + c * colq
    float32x4_t new_p = vsubq_f32(vmulq_f32(vc, colp), vmulq_f32(vs, colq));
    float32x4_t new_q = vaddq_f32(vmulq_f32(vs, colp), vmulq_f32(vc, colq));

    // Store back
    A[p*3]   = vgetq_lane_f32(new_p, 0);
    A[p*3+1] = vgetq_lane_f32(new_p, 1);
    A[p*3+2] = vgetq_lane_f32(new_p, 2);
    A[q*3]   = vgetq_lane_f32(new_q, 0);
    A[q*3+1] = vgetq_lane_f32(new_q, 1);
    A[q*3+2] = vgetq_lane_f32(new_q, 2);
}

//=============================================================================
// Compute A^T * A for 3x3 matrix (result is symmetric)
//=============================================================================
inline void compute_AtA_neon(const float* A, float* AtA) {
    // A is 3x3 column-major
    // AtA = A^T * A is symmetric, so we compute only upper triangle

    // Load columns of A
    float32x4_t a0 = {A[0], A[1], A[2], 0};
    float32x4_t a1 = {A[3], A[4], A[5], 0};
    float32x4_t a2 = {A[6], A[7], A[8], 0};

    // AtA(i, j) = dot(A(:,i), A(:,j))
    AtA[0] = vaddvq_f32_compat(vmulq_f32(a0, a0));  // (0,0)
    AtA[1] = AtA[3] = vaddvq_f32_compat(vmulq_f32(a0, a1));  // (0,1) = (1,0)
    AtA[2] = AtA[6] = vaddvq_f32_compat(vmulq_f32(a0, a2));  // (0,2) = (2,0)
    AtA[4] = vaddvq_f32_compat(vmulq_f32(a1, a1));  // (1,1)
    AtA[5] = AtA[7] = vaddvq_f32_compat(vmulq_f32(a1, a2));  // (1,2) = (2,1)
    AtA[8] = vaddvq_f32_compat(vmulq_f32(a2, a2));  // (2,2)
}

#endif // MNN_USE_NEON

} // namespace MNN::FaceGeometry::NEON



#endif  // MNN_FACE_GEOMETRY_NEON_INTRINSICS_H
