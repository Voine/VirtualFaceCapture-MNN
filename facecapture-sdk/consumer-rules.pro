# ===== facecapture-sdk consumer ProGuard rules =====
# Applied automatically to apps that consume this AAR.
# Keep JNI entry points and public data classes used across the JNI bridge.

# ---- Native bridge classes (MediaPipe face landmark MNN backend) ----
-keep class com.example.mediapipefacelandmark.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# ---- Public data types crossing module / JNI boundaries ----
-keep class com.example.commondata.** { *; }

# ---- Post-processing pipeline (OpenSeeFaceProcess) ----
# Note: historical package name is `com.live2d.facecapture` even though the
# module is Live2D-independent. Renaming is a future task.
-keep class com.live2d.facecapture.FaceCapturePipeline { *; }
-keep class com.live2d.facecapture.BlinkCurveProcessor { *; }
-keep class com.live2d.facecapture.TrackingConfidenceDetector { *; }
-keep class com.live2d.facecapture.SimpleEyeExtractor { *; }
-keep class com.live2d.facecapture.SimpleMouthExtractor { *; }
-keep class com.live2d.facecapture.SimpleSmileExtractor { *; }

# ---- SDK public surface ----
-keep class com.facecapture.sdk.** { *; }
