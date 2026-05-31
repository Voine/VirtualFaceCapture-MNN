# Face Geometry - No Dependency Version

MediaPipe Face Geometry 的无外部依赖 C++ 实现，专为移动端（Android/iOS）部署优化。

## 文件结构

### 核心文件（必需）

| 文件 | 行数 | 说明 |
|------|------|------|
| `mini_linalg.h` | ~830 | 轻量级线性代数库，替代 Eigen |
| `neon_intrinsics.h` | ~450 | ARM NEON SIMD 优化函数 |
| `procrustes_solver_nodep.h` | 320 | Procrustes 求解器 |
| `geometry_pipeline_nodep.h` | 700 | Face Geometry 主管线 |
| `canonical_face_mesh_data.h` | 545 | 硬编码的 468 点人脸网格数据 |

**总计：约 2800+ 行代码，全部 header-only，零外部依赖**

## NEON SIMD 优化

在 ARM 平台（Android/iOS）上自动启用 NEON 加速，优化以下热点函数：

### 已优化函数

| 函数 | 说明 | 预期加速 |
|------|------|----------|
| `VectorXf::sum()` | 向量求和 | 3-4x |
| `VectorXf::cwiseProduct()` | 向量逐元素乘法 | 3-4x |
| `Matrix3f::operator*` | 3x3 矩阵乘法 | 2x |
| `Matrix3Xf::cwiseProductSum()` | Frobenius 内积 | 3-4x |
| `Matrix3Xf::rowwiseProduct()` | 行向量乘法（468点×权重） | 3x |
| `Matrix3Xf::rowwiseSum()` | 行求和 | 3x |
| `Matrix3Xf::leftMultiply()` | 3x3 矩阵左乘 3xN | 2-3x |
| `Matrix3Xf::outerProduct()` | 外积（设计矩阵） | 2-3x |
| `Matrix3Xf::operator+/-` | 矩阵加减法 | 3-4x |
| `applyTransform()` | 4x4 齐次变换 | 2-3x |
| `computeSVD3x3()` | SVD 分解优化 | 1.5-2x |
| `jacobiRotateRight()` | Jacobi 旋转 | 1.5x |

### 编译选项

NEON 优化自动检测：
- 定义 `__ARM_NEON` 或 `__ARM_NEON__` 时启用
- ARM64 (`__aarch64__`) 使用原生 `vaddvq_f32`
- ARM32 使用兼容实现

如需强制禁用 NEON：
```cpp
#undef MNN_USE_NEON
#define MNN_USE_NEON 0
```

## 快速使用

```cpp
#include "geometry_pipeline_nodep.h"

using namespace MNN::FaceGeometry;

// 1. 创建 metadata（使用硬编码数据，无需文件 I/O）
auto metadata = CreateCanonicalFaceMeshMetadata();

// 2. 设置环境参数
Environment env;
env.origin_is_top_left = true;
env.vertical_fov_degrees = 63.0f;  // 相机垂直 FOV
env.near = 1.0f;
env.far = 10000.0f;

// 3. 创建 pipeline
auto pipeline = GeometryPipeline::Create(
    env,
    InputSource::FACE_LANDMARK_PIPELINE,
    metadata.canonical_mesh,
    metadata.landmark_weights);

// 4. 处理人脸 landmarks（来自 Face Landmark 模型的输出）
std::vector<NormalizedLandmark> face_landmarks(468);
// ... 填充 landmarks 数据 ...

std::vector<std::vector<NormalizedLandmark>> faces = {face_landmarks};
std::vector<FaceGeometry> results;

pipeline->EstimateFaceGeometry(faces, image_width, image_height, results);

// 5. 获取结果
if (!results.empty()) {
    // 4x4 头部姿态矩阵（旋转 + 平移）
    Matrix4f pose = results[0].pose_transform_matrix;
    
    // 3D 度量坐标系下的 landmarks（真实尺度，单位 cm）
    std::vector<Landmark3D> metric_landmarks = results[0].metric_landmarks;
}
```

## 输入格式

`NormalizedLandmark` 结构：
- `x`: 归一化 x 坐标 [0, 1]，左到右
- `y`: 归一化 y 坐标 [0, 1]，上到下
- `z`: 相对深度（可选，用于改善精度）

## 输出格式

`FaceGeometry` 结构：
- `pose_transform_matrix`: 4x4 变换矩阵，包含旋转和平移
- `metric_landmarks`: 468 个 3D 坐标点（真实尺度）

## 关于两个 pbtxt 文件

| 文件 | 顶点数 | 来源 | 推荐使用 |
|------|--------|------|----------|
| `geometry_pipeline_metadata_landmarks.pbtxt` | 468 | Face Landmark 模型 | ✅ 是 |
| `geometry_pipeline_metadata_detection.pbtxt` | 6 | Face Detection 模型 | ❌ 否 |

两者都能估计头部姿态，但 landmarks 版本精度高得多。硬编码版本 (`CreateCanonicalFaceMeshMetadata()`) 使用的是 landmarks 版本的数据。

## 编译测试

```bash
cd /path/to/MNN/build
clang++ -std=c++17 -I../demo/exec/face_geometry/libs \
    ../demo/exec/face_geometry/libs/test_face_geometry_nodep.cpp \
    -o test_face_geometry_nodep
./test_face_geometry_nodep
```

## 技术实现

### SVD 分解
使用 Jacobi 迭代法对称矩阵对角化，适用于 3x3 矩阵的 SVD 分解。

### Procrustes 分析
加权正交 Procrustes 问题求解，找到源点集到目标点集的最优刚性变换（旋转 + 缩放 + 平移）。

### 姿态估计流程
1. 将 2D landmarks 反投影到 3D 空间
2. 使用 Procrustes 分析对齐检测到的 landmarks 与 canonical mesh
3. 提取 4x4 变换矩阵和度量 3D landmarks

