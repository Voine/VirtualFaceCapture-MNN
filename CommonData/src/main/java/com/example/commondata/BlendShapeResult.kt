package com.example.commondata

/**
 * Author: Voine
 * Date: 2025/12/26
 * Description:
 */
/**
 * BlendShape result data class containing all 52 ARKit-compatible blendshapes
 *
 * This class wraps the raw float array from MediaPipe and provides convenient
 * access to common blendshapes. Use [toARKitBlendShapes] to convert to the
 * unified ARKitBlendShapes data class.
 */
data class BlendShapeResult(
    /** Raw 52-element float array in MediaPipe index order */
    val values: FloatArray,
    // Commonly used values for quick access
    val eyeBlinkLeft: Float,
    val eyeBlinkRight: Float,
    val jawOpen: Float,
    val mouthSmileLeft: Float,
    val mouthSmileRight: Float,
    val browInnerUp: Float,
    val browDownLeft: Float,
    val browDownRight: Float
) {
    /** Get blendshape value by index */
    fun get(index: Int): Float = values.getOrElse(index) { 0f }

    /**
     * Convert to unified ARKitBlendShapes data class
     * This is the recommended way to use the blendshape data
     */
    fun toARKitBlendShapes(): ARKitBlendShapes {
        return ARKitBlendShapes.fromMediaPipeArray(values)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as BlendShapeResult
        return values.contentEquals(other.values)
    }

    override fun hashCode(): Int = values.contentHashCode()
}
