/**
 * MNN-based YUV to RGB conversion with rotation
 * 
 * Uses MNN's optimized ImageProcess for hardware-accelerated conversion.
 * Supports rotation via affine transformation matrix.
 * 
 * Advantages over custom yuv_converter.cpp:
 * 1. MNN's ImageProcess is already SIMD optimized
 * 2. Unified with the rest of the MNN pipeline
 * 3. Supports affine transforms (rotation, scale) natively
 * 4. Can output directly to float tensor for inference
 */

#include <jni.h>
#include <android/log.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cmath>

#include "MNN/ImageProcess.hpp"
#include "MNN/Matrix.h"
#include "MNN/HalideRuntime.h"

#define LOG_TAG "MNNYuvConverter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

/**
 * Determine MNN ImageFormat from YUV_420_888 parameters
 * 
 * Android YUV_420_888 is a flexible format that can be:
 * - NV21 (Y + VU interleaved): uvPixelStride=2, V before U
 * - NV12 (Y + UV interleaved): uvPixelStride=2, U before V
 * - I420 (Y + U + V separate): uvPixelStride=1
 * 
 * @param uvPixelStride The pixel stride of U/V planes
 * @param uBuffer Pointer to U plane
 * @param vBuffer Pointer to V plane
 * @return MNN ImageFormat
 */
static MNN::CV::ImageFormat determineYuvFormat(int uvPixelStride, 
                                                const uint8_t* uBuffer, 
                                                const uint8_t* vBuffer) {
    if (uvPixelStride == 1) {
        // Planar format: Y, U, V are separate planes
        return MNN::CV::YUV_I420;
    } else if (uvPixelStride == 2) {
        // Semi-planar format: Y plane + interleaved UV or VU
        // Check if U comes before V in memory
        if (uBuffer < vBuffer) {
            // NV12: Y plane + UV interleaved
            return MNN::CV::YUV_NV12;
        } else {
            // NV21: Y plane + VU interleaved
            return MNN::CV::YUV_NV21;
        }
    }
    
    // Default to NV21 (most common on Android)
    LOGD("Unknown uvPixelStride=%d, defaulting to NV21", uvPixelStride);
    return MNN::CV::YUV_NV21;
}

/**
 * Create rotation matrix for MNN ImageProcess
 * 
 * MNN Matrix uses destination-to-source convention:
 * For each destination pixel, the matrix tells where to sample from source.
 * 
 * For a desired output rotation of R degrees clockwise, we need to apply
 * the INVERSE transformation (-R degrees) in the matrix.
 * 
 * @param rotation Rotation in degrees (0, 90, 180, 270) - this is the desired OUTPUT rotation
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @param dstWidth Destination image width  
 * @param dstHeight Destination image height
 * @return MNN Matrix for the transformation
 */
static MNN::CV::Matrix createRotationMatrix(int rotation,
                                             int srcWidth, int srcHeight,
                                             int dstWidth, int dstHeight) {
    MNN::CV::Matrix matrix;
    matrix.setIdentity();
    
    // MNN Matrix uses dst->src convention (inverse mapping)
    // We need to map destination coordinates to source coordinates
    // So we apply the INVERSE of the desired transformation
    
    float cx_dst = static_cast<float>(dstWidth) / 2.0f;
    float cy_dst = static_cast<float>(dstHeight) / 2.0f;
    float cx_src = static_cast<float>(srcWidth) / 2.0f;
    float cy_src = static_cast<float>(srcHeight) / 2.0f;
    
    // The inverse of rotation by R is rotation by -R
    // postTranslate/postRotate chain: first translate dst to origin,
    // then rotate back (negative), then translate to src center
    
    switch (rotation) {
        case 0:
            // No rotation, identity matrix works
            break;
            
        case 90:
            // To rotate output 90° CW, we sample from src with 90° CCW (i.e., -90°)
            // dst coords -> translate to origin -> rotate -90° -> translate to src center
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(-90.0f);
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        case 180:
            // 180° rotation (same as -180°)
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(180.0f);  // 180° == -180°
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        case 270:
            // To rotate output 270° CW (= 90° CCW), we sample with +90° rotation
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(90.0f);  // Inverse of 270° is -270° = 90°
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        default:
            LOGE("Unsupported rotation: %d, using 0", rotation);
            break;
    }
    
    return matrix;
}

/**
 * Prepare YUV data buffer for MNN ImageProcess
 * 
 * MNN expects YUV data in a contiguous buffer:
 * - For NV21/NV12: Y plane followed by interleaved UV/VU
 * - For I420: Y plane, U plane, V plane
 * 
 * @param yData Y plane data
 * @param uData U plane data  
 * @param vData V plane data
 * @param yRowStride Y plane row stride
 * @param uvRowStride UV plane row stride
 * @param uvPixelStride UV pixel stride
 * @param width Image width
 * @param height Image height
 * @param format Detected YUV format
 * @param outBuffer Output buffer (must be pre-allocated)
 * @return Size of data written to outBuffer
 */
static size_t prepareYuvBuffer(const uint8_t* yData,
                               const uint8_t* uData,
                               const uint8_t* vData,
                               int yRowStride,
                               int uvRowStride,
                               int uvPixelStride,
                               int width, int height,
                               MNN::CV::ImageFormat format,
                               uint8_t* outBuffer) {
    size_t offset = 0;
    
    // Copy Y plane (handle stride)
    if (yRowStride == width) {
        // Contiguous, direct copy
        memcpy(outBuffer, yData, width * height);
        offset = width * height;
    } else {
        // Need to remove padding
        for (int y = 0; y < height; y++) {
            memcpy(outBuffer + y * width, yData + y * yRowStride, width);
        }
        offset = width * height;
    }
    
    int uvHeight = height / 2;
    int uvWidth = width / 2;
    
    if (format == MNN::CV::YUV_I420) {
        // I420: Copy U plane, then V plane
        if (uvPixelStride == 1 && uvRowStride == uvWidth) {
            // Contiguous U and V planes
            memcpy(outBuffer + offset, uData, uvWidth * uvHeight);
            offset += uvWidth * uvHeight;
            memcpy(outBuffer + offset, vData, uvWidth * uvHeight);
            offset += uvWidth * uvHeight;
        } else {
            // Need to extract with stride
            for (int y = 0; y < uvHeight; y++) {
                for (int x = 0; x < uvWidth; x++) {
                    outBuffer[offset++] = uData[y * uvRowStride + x * uvPixelStride];
                }
            }
            for (int y = 0; y < uvHeight; y++) {
                for (int x = 0; x < uvWidth; x++) {
                    outBuffer[offset++] = vData[y * uvRowStride + x * uvPixelStride];
                }
            }
        }
    } else {
        // NV12 or NV21: Interleaved UV plane
        // For NV12: UVUVUV...
        // For NV21: VUVUVU...
        // Since we detected format from buffer order, just copy the interleaved data
        const uint8_t* uvStart = (format == MNN::CV::YUV_NV12) ? uData : vData;
        
        if (uvRowStride == width) {
            // Contiguous interleaved UV
            memcpy(outBuffer + offset, uvStart, width * uvHeight);
            offset += width * uvHeight;
        } else {
            // Handle stride
            for (int y = 0; y < uvHeight; y++) {
                memcpy(outBuffer + offset + y * width, uvStart + y * uvRowStride, width);
            }
            offset += width * uvHeight;
        }
    }
    
    return offset;
}

/**
 * Convert YUV420 to RGB with rotation using MNN ImageProcess
 * 
 * This is the main entry point for MNN-based conversion.
 * 
 * @param yData Y plane data pointer
 * @param uData U plane data pointer
 * @param vData V plane data pointer
 * @param yRowStride Y plane row stride
 * @param uvRowStride UV plane row stride
 * @param uvPixelStride UV pixel stride (1=I420, 2=NV12/NV21)
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @param rotation Rotation in degrees (0, 90, 180, 270)
 * @param rgbOut Output RGB buffer (must be pre-allocated: dstWidth * dstHeight * 3)
 * @param dstWidth Output width after rotation
 * @param dstHeight Output height after rotation
 */
extern "C" void convertYuvToRgbMNN(
    const uint8_t* yData,
    const uint8_t* uData,
    const uint8_t* vData,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride,
    int srcWidth,
    int srcHeight,
    int rotation,
    uint8_t* rgbOut,
    int dstWidth,
    int dstHeight
) {
    // Determine YUV format
    MNN::CV::ImageFormat yuvFormat = determineYuvFormat(uvPixelStride, uData, vData);
    
    // Prepare contiguous YUV buffer
    size_t yuvBufferSize = srcWidth * srcHeight + srcWidth * srcHeight / 2;
    std::unique_ptr<uint8_t[]> yuvBuffer(new uint8_t[yuvBufferSize]);
    
    prepareYuvBuffer(yData, uData, vData, yRowStride, uvRowStride, uvPixelStride,
                     srcWidth, srcHeight, yuvFormat, yuvBuffer.get());
    
    // Configure MNN ImageProcess
    MNN::CV::ImageProcess::Config config;
    config.sourceFormat = yuvFormat;
    config.destFormat = MNN::CV::RGB;
    config.filterType = MNN::CV::BILINEAR;  // Smooth interpolation
    config.wrap = MNN::CV::ZERO;            // Black padding for out-of-bounds
    
    // No normalization for uint8 output
    config.mean[0] = 0.0f;
    config.mean[1] = 0.0f;
    config.mean[2] = 0.0f;
    config.normal[0] = 1.0f;
    config.normal[1] = 1.0f;
    config.normal[2] = 1.0f;
    
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    
    // Set rotation matrix
    if (rotation != 0) {
        MNN::CV::Matrix matrix = createRotationMatrix(rotation, srcWidth, srcHeight, dstWidth, dstHeight);
        process->setMatrix(matrix);
    }
    
    // Perform conversion
    // Using raw pointer output (simpler than tensor for just RGB bytes)
    process->convert(yuvBuffer.get(), srcWidth, srcHeight, 0,
                     rgbOut, dstWidth, dstHeight, 3, 0,
                     halide_type_t{halide_type_uint, 8, 1});
}

// ============================================================================
// JNI Interface
// ============================================================================

extern "C" JNIEXPORT void JNICALL
Java_com_example_mediapipefacelandmark_MNNYuvConverter_nativeConvertYuvToRgb(
    JNIEnv* env,
    jobject thiz,
    jobject yBuffer,
    jobject uBuffer,
    jobject vBuffer,
    jint yRowStride,
    jint uvRowStride,
    jint uvPixelStride,
    jint srcWidth,
    jint srcHeight,
    jint rotation,
    jobject rgbBuffer,
    jint dstWidth,
    jint dstHeight
) {
    // Get direct buffer addresses
    auto yData = static_cast<uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto uData = static_cast<uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto vData = static_cast<uint8_t*>(env->GetDirectBufferAddress(vBuffer));
    auto rgbOut = static_cast<uint8_t*>(env->GetDirectBufferAddress(rgbBuffer));
    
    if (!yData || !uData || !vData || !rgbOut) {
        LOGE("Failed to get direct buffer addresses");
        return;
    }
    
    convertYuvToRgbMNN(
        yData, uData, vData,
        yRowStride, uvRowStride, uvPixelStride,
        srcWidth, srcHeight, rotation,
        rgbOut, dstWidth, dstHeight
    );
}

/**
 * Check if MNN YUV converter is available
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_mediapipefacelandmark_MNNYuvConverter_nativeIsAvailable(
    JNIEnv* /* env */,
    jobject thiz) {
    return JNI_TRUE;
}
