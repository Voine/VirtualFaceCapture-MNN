package com.example.commondata

/**
 * Data class for face detection visualization
 * Contains all data needed to render face landmarks and detection info on camera preview
 */
data class FaceVisualizationData(
    // Landmarks (normalized 0-1 coordinates)
    val landmarks: List<Point3D>,
    
    // Detection box (normalized coordinates)
    val detectionBox: DetectionBox?,
    
    // Head pose angles (degrees)
    val headPose: HeadPoseData,
    
    // Image dimensions
    val imageWidth: Int,
    val imageHeight: Int,
    
    // Timestamp for tracking
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Detection bounding box (normalized 0-1 coordinates)
 */
data class DetectionBox(
    val x1: Float,  // left
    val y1: Float,  // top
    val x2: Float,  // right
    val y2: Float,  // bottom
    val score: Float = 0f
)

/**
 * Head pose data containing Euler angles and translation
 */
data class HeadPoseData(
    val pitch: Float = 0f,  // up/down rotation (degrees)
    val yaw: Float = 0f,    // left/right rotation (degrees)
    val roll: Float = 0f,   // tilt rotation (degrees)
    val translationX: Float = 0f,
    val translationY: Float = 0f,
    val translationZ: Float = 0f
) {
    val isValid: Boolean get() = pitch != 0f || yaw != 0f || roll != 0f
    
    override fun toString(): String {
        return String.format("pitch=%.1f° yaw=%.1f° roll=%.1f°", pitch, yaw, roll)
    }
}
