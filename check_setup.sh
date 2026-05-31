#!/bin/bash
# Setup verification script for Live2D Face Pipeline

echo "=================================="
echo "Live2D Face Pipeline Setup Checker"
echo "=================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check function
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2 (NOT FOUND: $1)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2 (NOT FOUND: $1)"
        return 1
    fi
}

ERRORS=0

echo "1. Checking MediapipeFaceLandmark module..."
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/assets/face_detector_mp.mnn" "Face detector model" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/assets/face_landmark_detector_mp.mnn" "Face landmarker model" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/cpp/mediapipefacelandmark.cpp" "JNI bridge implementation" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/cpp/simple_mediapipe_face.cpp" "MediaPipe face detector" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/java/com/example/mediapipefacelandmark/Camera2Manager.kt" "Camera2 manager" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/java/com/example/mediapipefacelandmark/MediaPipeFaceDetector.kt" "MediaPipe face detector wrapper" || ((ERRORS++))
echo ""

echo "2. Checking OpenSeeFaceProcess module..."
check_file "$PROJECT_ROOT/OpenSeeFaceProcess/src/main/java/com/live2d/facecapture/FaceCapturePipeline.kt" "Face capture pipeline" || ((ERRORS++))
check_file "$PROJECT_ROOT/OpenSeeFaceProcess/src/main/java/com/live2d/facecapture/BlendShapeExtractor.kt" "BlendShape extractor" || ((ERRORS++))
check_file "$PROJECT_ROOT/OpenSeeFaceProcess/src/main/java/com/live2d/facecapture/Landmark.kt" "Landmark definitions" || ((ERRORS++))
echo ""

echo "3. Checking Live2D module..."
check_dir "$PROJECT_ROOT/Live2D/src/main/assets/Live2DModels" "Live2D models directory" || ((ERRORS++))
check_file "$PROJECT_ROOT/Live2D/src/main/java/com/live2d/demo/JniBridgeJava.java" "Live2D JNI bridge" || ((ERRORS++))
check_file "$PROJECT_ROOT/Live2D/src/main/java/com/live2d/demo/GLRenderer.java" "Live2D GL renderer" || ((ERRORS++))

# Count Live2D models
MODEL_COUNT=$(ls -d "$PROJECT_ROOT/Live2D/src/main/assets/Live2DModels"/*/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Found $MODEL_COUNT Live2D model(s)"
    ls -d "$PROJECT_ROOT/Live2D/src/main/assets/Live2DModels"/*/ 2>/dev/null | while read model; do
        echo "    - $(basename "$model")"
    done
else
    echo -e "${RED}✗${NC} No Live2D models found"
    ((ERRORS++))
fi
echo ""

echo "4. Checking App module..."
check_file "$PROJECT_ROOT/app/src/main/java/com/example/live2dfacepipeline/MainActivity.kt" "Main activity" || ((ERRORS++))
check_file "$PROJECT_ROOT/app/src/main/java/com/example/live2dfacepipeline/FaceTracker.kt" "Face tracker" || ((ERRORS++))
check_file "$PROJECT_ROOT/app/src/main/java/com/example/live2dfacepipeline/Live2DView.kt" "Live2D view" || ((ERRORS++))
check_file "$PROJECT_ROOT/app/src/main/AndroidManifest.xml" "App manifest" || ((ERRORS++))

# Check camera permission in manifest
if grep -q "android.permission.CAMERA" "$PROJECT_ROOT/app/src/main/AndroidManifest.xml" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Camera permission declared in manifest"
else
    echo -e "${RED}✗${NC} Camera permission NOT found in manifest"
    ((ERRORS++))
fi
echo ""

echo "5. Checking build configuration..."
check_file "$PROJECT_ROOT/app/build.gradle" "App build.gradle" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/build.gradle" "MediapipeFaceLandmark build.gradle" || ((ERRORS++))
check_file "$PROJECT_ROOT/MediapipeFaceLandmark/src/main/cpp/CMakeLists.txt" "CMakeLists.txt" || ((ERRORS++))

# Check module dependencies in app/build.gradle
if grep -q "project(':MediapipeFaceLandmark')" "$PROJECT_ROOT/app/build.gradle" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} MediapipeFaceLandmark dependency in app"
else
    echo -e "${YELLOW}⚠${NC} MediapipeFaceLandmark dependency may be missing"
fi

if grep -q "project(':OpenSeeFaceProcess')" "$PROJECT_ROOT/app/build.gradle" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} OpenSeeFaceProcess dependency in app"
else
    echo -e "${YELLOW}⚠${NC} OpenSeeFaceProcess dependency may be missing"
fi

if grep -q "project(':Live2D')" "$PROJECT_ROOT/app/build.gradle" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Live2D dependency in app"
else
    echo -e "${YELLOW}⚠${NC} Live2D dependency may be missing"
fi
echo ""

echo "=================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "Project is ready to build."
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo "Please fix the issues above before building."
    exit 1
fi
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Open project in Android Studio"
echo "  2. Sync Gradle"
echo "  3. Build project: ./gradlew build"
echo "  4. Run on device with camera"
echo ""
echo "For more information, see IMPLEMENTATION_GUIDE.md"
