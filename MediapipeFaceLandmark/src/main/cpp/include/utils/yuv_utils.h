/**
 * @file yuv_utils.h
 * @brief YUV image processing utilities
 * 
 * Provides utilities for handling YUV image data from Android Camera2 API,
 * including format detection, buffer preparation, and rotation matrices.
 */

#ifndef MEDIAPIPE_FACE_YUV_UTILS_H
#define MEDIAPIPE_FACE_YUV_UTILS_H

#include <cstdint>
#include <vector>
#include <cstring>
#include "MNN/ImageProcess.hpp"
#include "MNN/Matrix.h"
#include "common/logging.h"

namespace mediapipe_face {
namespace yuv {

/**
 * @brief Determine MNN ImageFormat from YUV_420_888 parameters
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
inline MNN::CV::ImageFormat detectFormat(int uvPixelStride, 
                                         const uint8_t* uBuffer, 
                                         const uint8_t* vBuffer) {
    if (uvPixelStride == 1) {
        return MNN::CV::YUV_I420;
    } else if (uvPixelStride == 2) {
        // Check if U comes before V in memory
        if (uBuffer < vBuffer) {
            return MNN::CV::YUV_NV12;
        } else {
            return MNN::CV::YUV_NV21;
        }
    }
    
    FACE_LOGW("Unknown uvPixelStride=%d, defaulting to NV21", uvPixelStride);
    return MNN::CV::YUV_NV21;
}

/**
 * @brief Prepare YUV data into contiguous buffer for MNN ImageProcess
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
 * @param outBuffer Output buffer (will be resized)
 * @return Size of data written to outBuffer
 */
inline size_t prepareBuffer(const uint8_t* yData,
                            const uint8_t* uData,
                            const uint8_t* vData,
                            int yRowStride,
                            int uvRowStride,
                            int uvPixelStride,
                            int width, int height,
                            MNN::CV::ImageFormat format,
                            std::vector<uint8_t>& outBuffer) {
    // Calculate buffer size: Y plane + UV plane (half height)
    size_t ySize = width * height;
    size_t uvSize = width * height / 2;
    outBuffer.resize(ySize + uvSize);
    
    size_t offset = 0;
    
    // Copy Y plane (handle stride)
    if (yRowStride == width) {
        memcpy(outBuffer.data(), yData, ySize);
        offset = ySize;
    } else {
        for (int y = 0; y < height; y++) {
            memcpy(outBuffer.data() + y * width, yData + y * yRowStride, width);
        }
        offset = ySize;
    }
    
    int uvHeight = height / 2;
    int uvWidth = width / 2;
    
    if (format == MNN::CV::YUV_I420) {
        // I420: Copy U plane, then V plane
        if (uvPixelStride == 1 && uvRowStride == uvWidth) {
            memcpy(outBuffer.data() + offset, uData, uvWidth * uvHeight);
            offset += uvWidth * uvHeight;
            memcpy(outBuffer.data() + offset, vData, uvWidth * uvHeight);
            offset += uvWidth * uvHeight;
        } else {
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
        const uint8_t* uvStart = (format == MNN::CV::YUV_NV12) ? uData : vData;
        if (uvRowStride == width) {
            memcpy(outBuffer.data() + offset, uvStart, width * uvHeight);
            offset += width * uvHeight;
        } else {
            for (int y = 0; y < uvHeight; y++) {
                memcpy(outBuffer.data() + offset + y * width, uvStart + y * uvRowStride, width);
            }
            offset += width * uvHeight;
        }
    }
    
    return offset;
}

/**
 * @brief Create rotation matrix for MNN ImageProcess
 * 
 * MNN Matrix uses destination-to-source convention:
 * For each destination pixel, the matrix tells where to sample from source.
 * 
 * @param rotation Rotation in degrees (0, 90, 180, 270) - desired OUTPUT rotation
 * @param srcWidth Source image width
 * @param srcHeight Source image height
 * @param dstWidth Destination image width  
 * @param dstHeight Destination image height
 * @return MNN Matrix for the transformation
 */
inline MNN::CV::Matrix createRotationMatrix(int rotation,
                                             int srcWidth, int srcHeight,
                                             int dstWidth, int dstHeight) {
    MNN::CV::Matrix matrix;
    matrix.setIdentity();
    
    float cx_dst = static_cast<float>(dstWidth) / 2.0f;
    float cy_dst = static_cast<float>(dstHeight) / 2.0f;
    float cx_src = static_cast<float>(srcWidth) / 2.0f;
    float cy_src = static_cast<float>(srcHeight) / 2.0f;
    
    // For dst->src convention, we apply inverse rotation
    switch (rotation) {
        case 0:
            break;
            
        case 90:
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(-90.0f);
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        case 180:
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(180.0f);
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        case 270:
            matrix.postTranslate(-cx_dst, -cy_dst);
            matrix.postRotate(90.0f);
            matrix.postTranslate(cx_src, cy_src);
            break;
            
        default:
            FACE_LOGE("Unsupported rotation: %d, using 0", rotation);
            break;
    }
    
    return matrix;
}

/**
 * @brief Get output dimensions after rotation
 * 
 * @param srcWidth Source width
 * @param srcHeight Source height
 * @param rotation Rotation in degrees
 * @param outWidth Output width (will be set)
 * @param outHeight Output height (will be set)
 */
inline void getRotatedDimensions(int srcWidth, int srcHeight, int rotation,
                                  int& outWidth, int& outHeight) {
    if (rotation == 90 || rotation == 270) {
        outWidth = srcHeight;
        outHeight = srcWidth;
    } else {
        outWidth = srcWidth;
        outHeight = srcHeight;
    }
}

} // namespace yuv
} // namespace mediapipe_face

#endif // MEDIAPIPE_FACE_YUV_UTILS_H
