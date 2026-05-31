package com.example.mediapipefacelandmark

import android.util.Log
import java.nio.ByteBuffer

/**
 * Native YUV to RGB converter using C++ with optional NEON SIMD optimization
 * Much faster than pure Kotlin implementation
 */
object YuvConverter {

    private const val TAG = "YuvConverter"

    private var isNativeLoaded = false

    init {
        try {
            System.loadLibrary("mediapipefacelandmark")
            isNativeLoaded = true
            Log.i(TAG, "Native library loaded. NEON support: ${nativeHasNeonSupport()}")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library", e)
            isNativeLoaded = false
        }
    }

    /**
     * Check if native conversion is available
     */
    fun isNativeAvailable(): Boolean = isNativeLoaded

    /**
     * Convert YUV420 to RGB with rotation using native code
     *
     * @param yBuffer Y plane ByteBuffer (must be direct)
     * @param uBuffer U plane ByteBuffer (must be direct)
     * @param vBuffer V plane ByteBuffer (must be direct)
     * @param yRowStride Y plane row stride
     * @param uvRowStride UV plane row stride
     * @param uvPixelStride UV plane pixel stride
     * @param srcWidth Source image width
     * @param srcHeight Source image height
     * @param rotation Rotation in degrees (0, 90, 180, 270)
     * @param rgbBuffer Output RGB buffer (must be direct, pre-allocated with size outWidth * outHeight * 3)
     * @param outWidth Output width after rotation
     * @param outHeight Output height after rotation
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
        outWidth: Int,
        outHeight: Int
    ) {
        if (!isNativeLoaded) {
            throw IllegalStateException("Native library not loaded")
        }

        nativeConvertYuvToRgb(
            yBuffer, uBuffer, vBuffer,
            yRowStride, uvRowStride, uvPixelStride,
            srcWidth, srcHeight, rotation,
            rgbBuffer, outWidth, outHeight
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
        outWidth: Int,
        outHeight: Int
    )

    external fun nativeHasNeonSupport(): Boolean
}

