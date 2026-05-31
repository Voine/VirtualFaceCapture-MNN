package com.example.mediapipefacelandmark

import android.util.Log
import java.nio.ByteBuffer

/**
 * MNN-based YUV to RGB converter
 * 
 * Uses MNN's optimized ImageProcess for hardware-accelerated YUV->RGB conversion
 * with rotation support via affine transformation matrix.
 * 
 * Advantages over custom yuv_converter:
 * 1. MNN's ImageProcess is already SIMD optimized
 * 2. Unified with the rest of the MNN inference pipeline
 * 3. Supports affine transforms (rotation, scale) natively
 * 4. Can potentially output directly to float tensor for inference
 */
object MNNYuvConverter {
    
    private const val TAG = "MNNYuvConverter"
    
    private var isNativeLoaded = false
    
    init {
        try {
            // The native library should already be loaded by MediaPipeFaceDetector
            // This just verifies MNN conversion is available
            System.loadLibrary("mediapipefacelandmark")
            isNativeLoaded = nativeIsAvailable()
            Log.i(TAG, "MNN YUV converter loaded: $isNativeLoaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library", e)
            isNativeLoaded = false
        }
    }
    
    /**
     * Check if MNN-based conversion is available
     */
    fun isAvailable(): Boolean = isNativeLoaded
    
    /**
     * Convert YUV420 to RGB with rotation using MNN ImageProcess
     * 
     * @param yBuffer Y plane ByteBuffer (must be direct)
     * @param uBuffer U plane ByteBuffer (must be direct)
     * @param vBuffer V plane ByteBuffer (must be direct)
     * @param yRowStride Y plane row stride
     * @param uvRowStride UV plane row stride
     * @param uvPixelStride UV plane pixel stride (1=I420, 2=NV12/NV21)
     * @param srcWidth Source image width
     * @param srcHeight Source image height
     * @param rotation Rotation in degrees (0, 90, 180, 270)
     * @param rgbBuffer Output RGB buffer (must be direct, pre-allocated with size dstWidth * dstHeight * 3)
     * @param dstWidth Output width after rotation
     * @param dstHeight Output height after rotation
     */
    fun convertYuvToRgb(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        rgbBuffer: ByteBuffer,
        dstWidth: Int,
        dstHeight: Int
    ) {
        if (!isNativeLoaded) {
            throw IllegalStateException("MNN YUV converter not available")
        }
        
        require(yBuffer.isDirect) { "yBuffer must be a direct ByteBuffer" }
        require(uBuffer.isDirect) { "uBuffer must be a direct ByteBuffer" }
        require(vBuffer.isDirect) { "vBuffer must be a direct ByteBuffer" }
        require(rgbBuffer.isDirect) { "rgbBuffer must be a direct ByteBuffer" }
        
        val requiredSize = dstWidth * dstHeight * 3
        require(rgbBuffer.capacity() >= requiredSize) { 
            "rgbBuffer too small: need $requiredSize, got ${rgbBuffer.capacity()}" 
        }
        
        nativeConvertYuvToRgb(
            yBuffer, uBuffer, vBuffer,
            yRowStride, uvRowStride, uvPixelStride,
            srcWidth, srcHeight, rotation,
            rgbBuffer, dstWidth, dstHeight
        )
    }
    
    // Native methods
    private external fun nativeConvertYuvToRgb(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        rgbBuffer: ByteBuffer,
        dstWidth: Int,
        dstHeight: Int
    )
    
    private external fun nativeIsAvailable(): Boolean
}
