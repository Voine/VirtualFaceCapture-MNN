# facecapture-sdk

> Standalone **face-capture AAR** extracted from VirtualFaceCapture-MNN.
> Pure face-tracking stack: **MediaPipe topology + MNN inference + OpenSeeFace-style post-processing**.
> No Live2D, no UI, no app code.

---

## 1. What this AAR contains

| Layer                    | Provides                                                                                | Source module               |
|--------------------------|-----------------------------------------------------------------------------------------|-----------------------------|
| Native inference         | `libMNN.so`, `libMNN_Express.so`, `libmediapipefacelandmark.so`, `face_*.mnn` models    | `:MediapipeFaceLandmark`    |
| Post-processing          | ARKit-52 BlendShape extraction, head-pose recovery, blink-curve, smoothing, calibration | `:OpenSeeFaceProcess`       |
| Shared data types        | `Point3D`, `ARKitBlendShapes`, `HeadPose`, `FaceKeyPoints`, `FaceLandmarkIndices`, …    | `:CommonData`               |
| Public facade            | `FaceCaptureEngine` interface + `FaceCaptureConfig` + `CameraFrame` + `FaceCaptureResult` | `:facecapture-sdk` (this)   |

What it intentionally **does not** ship:
- Live2D Cubism renderer (lives in `:Live2D`).
- Camera2/CameraX wiring (`Camera2Manager` is currently in the inference
  module but the SDK contract is camera-agnostic — see §4).
- ARKit→Live2D parameter mapping (UI/demo code, lives in `:app`).

---

## 2. Module dependency graph (proves it's clean to extract)

```
            ┌────────────────────────┐
            │   facecapture-sdk      │  ← published AAR (facade)
            └──────────┬─────────────┘
                       │ api
   ┌───────────────────┼───────────────────────┐
   ▼                   ▼                       ▼
MediapipeFaceLandmark  OpenSeeFaceProcess     CommonData
   │                   │                       ▲
   │ implementation    │ implementation        │
   └���──────────────────┴───────────────────────┘
                       (DAG, single source of truth)

Live2D, app  ──►  NEVER referenced by any of the four boxes above.
```

Verified with `grep` — there are **zero** imports from any face-capture
module pointing back at `Live2D` or `com.example.virtualfacecapture`.

---

## 3. Public API (Kotlin contract)

See [`FaceCaptureEngine.kt`](src/main/java/com/facecapture/sdk/FaceCaptureEngine.kt) for KDoc.

```kotlin
val engine = FaceCaptureEngine.create()
engine.initialize(context)                          // loads MNN models from AAR assets
engine.setResultListener { result ->
    val landmarks: List<Point3D>     = result.landmarks        // 478 raw pts
    val blendShapes: ARKitBlendShapes = result.blendShapes     // 52 BS, smoothed
    val pose: HeadPose                = result.relativeHeadPose
    // forward to your avatar / network / file
}

// From CameraX ImageAnalysis.Analyzer:
override fun analyze(image: ImageProxy) {
    image.image?.let { engine.pushFrame(it, image.imageInfo.rotationDegrees) }
    image.close()
}

// On teardown:
engine.release()
```

Output objects (all in `:CommonData`, re-exported by the facade via `api`):

| Type                 | Meaning                                                                              |
|----------------------|--------------------------------------------------------------------------------------|
| `Point3D`            | Single MediaPipe landmark in image space.                                            |
| `ARKitBlendShapes`   | All 52 ARKit BlendShape values (`eyeBlinkLeft`, `mouthSmileLeft`, `jawOpen`, …).     |
| `HeadPose`           | `pitch / yaw / roll` in degrees.                                                     |
| `FaceCaptureResult`  | Wraps the above + presence score + calibration flag + rolling FPS for one frame.     |

---

## 4. Input model: **camera-agnostic by design**

The user request was *"accept a camera texture, emit landmark array + post-processed data"*.
Two equivalent input paths are exposed:

1. **`pushFrame(image: android.media.Image, rotationDegrees: Int)`** — drop-in for
   CameraX `ImageAnalysis` or Camera2 `ImageReader`. The SDK extracts the YUV
   planes internally. Recommended for most apps.

2. **`pushFrame(frame: CameraFrame)`** — zero-copy path that mirrors
   `ImageFormat.YUV_420_888` plane layout. Use this when integrating with a
   custom capture stack (Unity native bridge, Flutter platform channels, etc.).

> Pure **SurfaceTexture / GL texture** input is *not* in v0.1.0 — MNN consumes
> CPU tensors, so a GL texture would round-trip through `glReadPixels` anyway.
> If you have a GL texture, the cheapest route is to feed CameraX an
> `ImageAnalysis` use-case in parallel with your `Preview` use-case.

---

## 5. Build & publish

### 5.1 Build the AAR locally

```bash
./gradlew :facecapture-sdk:assembleRelease
# Output: facecapture-sdk/build/outputs/aar/facecapture-sdk-release.aar
```

This produces **one** AAR for the facade, but as a normal multi-module Android
library that one AAR has `pom`-level dependencies on the three upstream AARs.
For a local maven dump:

```bash
./gradlew :facecapture-sdk:publishReleasePublicationToLocalRepoRepository \
          :MediapipeFaceLandmark:publishToMavenLocal \
          :OpenSeeFaceProcess:publishToMavenLocal \
          :CommonData:publishToMavenLocal
# Output: facecapture-sdk/build/repo/io/github/virtualfacecapture/facecapture-mnn/0.1.0/
```

(`:MediapipeFaceLandmark`, `:OpenSeeFaceProcess`, `:CommonData` need their own
`maven-publish` blocks if you want to publish them separately — currently only
the facade declares one. The simplest path is option **5.2** below.)

### 5.2 Distribute as **one fat AAR** (recommended for closed-source release)

Merge the four AARs into one self-contained file via
[`kezong/fat-aar-android`](https://github.com/kezong/fat-aar-android):

1. Root `build.gradle`:
   ```groovy
   buildscript {
       dependencies {
           classpath 'com.kezong:fat-aar:1.3.8'   // pick the latest version
       }
   }
   ```
2. In `facecapture-sdk/build.gradle`, at the top:
   ```groovy
   apply plugin: 'com.kezong.fat-aar'
   ```
3. Replace the `api project(...)` lines with `embed project(...)`:
   ```groovy
   dependencies {
       embed project(':MediapipeFaceLandmark')
       embed project(':OpenSeeFaceProcess')
       embed project(':CommonData')
   }
   ```
4. Re-run `./gradlew :facecapture-sdk:assembleRelease`. The output AAR now
   contains all `.so` files, `.mnn` models, classes, and resources from the
   three modules in a single artifact.

Trade-off: `fat-aar` is a third-party plugin and occasionally lags behind new
AGP / Kotlin releases. For open-source / maven-central style publishing, the
default multi-AAR approach is more idiomatic.

### 5.3 Maven coordinates

Configured in `facecapture-sdk/build.gradle`:

```
group:    io.github.virtualfacecapture
artifact: facecapture-mnn
version:  0.1.0
```

Change `sdkGroupId` / `sdkArtifactId` / `sdkVersion` (top of the gradle file)
to match your own publishing domain before pushing to GitHub Packages,
JitPack, or Sonatype.

---

## 6. Consumer integration (downstream app)

```groovy
// settings.gradle of the consumer app
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        // for option 5.1 with local repo:
        // maven { url 'file:///path/to/facecapture-sdk/build/repo' }
        // for GitHub Packages:
        // maven { url 'https://maven.pkg.github.com/<owner>/VirtualFaceCapture-MNN' }
    }
}

// app/build.gradle of the consumer
android {
    defaultConfig {
        ndk { abiFilters 'arm64-v8a' }   // SDK currently ships arm64-v8a only
    }
    packaging {
        // Only needed if the consumer also bundles its own libMNN.so:
        // jniLibs.pickFirsts += '**/libMNN.so'
        // jniLibs.pickFirsts += '**/libMNN_Express.so'
    }
}

dependencies {
    implementation 'io.github.virtualfacecapture:facecapture-mnn:0.1.0'
    implementation 'androidx.camera:camera-camera2:1.3.4'
    implementation 'androidx.camera:camera-lifecycle:1.3.4'
}
```

Runtime permissions still required by the consumer: `android.permission.CAMERA`.
The SDK itself does **not** request the permission.

---

## 7. Migrating `PipelineFaceTracker` into the SDK

The reference implementation backing `FaceCaptureEngine.create()` is the
`PipelineFaceTracker` class currently sitting in
`app/src/main/java/com/example/virtualfacecapture/PipelineFaceTracker.kt`.
It already wraps everything we want, but it also reaches into Live2D
(`JniBridgeJava`) and into demo-only data classes (`HeadPoseData`,
`FaceVisualizationData`). The clean extraction is:

1. **Move data classes** to `:CommonData` (or to this SDK):
   - `HeadPoseData`        → `com.example.commondata.HeadPose` (already exists; alias/merge).
   - `FaceVisualizationData` → keep as `internal` to the SDK; it's a debug helper.
2. **Copy `PipelineFaceTracker.kt`** to
   `facecapture-sdk/src/main/java/com/facecapture/sdk/internal/FaceCaptureEngineImpl.kt`.
3. **Delete** these blocks from the copy:
   - `import com.live2d.demo.JniBridgeJava`
   - `applyBlendShapesToLive2D(...)` and every call site.
   - Any `onCameraFrameUpdate` / `onVisualizationUpdate` callbacks that are
     purely for the in-app preview UI (optional — they can stay if you want
     to expose a debug stream).
4. **Adapt the input surface**:
   - Replace the private `Camera2Manager` ownership with the public
     `pushFrame(...)` channel defined in `FaceCaptureEngine`.
   - The existing `detectFaceFromYuv(...)` path inside `MediapipeFaceLandmark`
     already accepts the exact `YuvFrameData` shape, so this is mostly a
     plumbing change.
5. **Wire the factory**: replace the `NotImplementedError` in
   `FaceCaptureEngine.create()` with `return FaceCaptureEngineImpl()`.
6. **Original `app/`** keeps working unchanged — it can either continue using
   its own `PipelineFaceTracker` (for Live2D-specific glue) or migrate to
   `FaceCaptureEngine` + a thin Live2D adapter. Either is non-breaking.

Estimated effort: **1–2 person-days**.

---

## 8. Native packaging notes

- `MediapipeFaceLandmark/src/main/cpp/CMakeLists.txt` already supports
  "consumed as sub-module" vs "built standalone" via
  `CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR`. When built through
  the SDK the sub-module branch is used, exporting a static `mediapipe_face_core`
  library and the headers (now pointing at `ThirdParty/MNN/include/MNN`).
- Pre-built `.so` files in `MediapipeFaceLandmark/src/main/jniLibs/arm64-v8a/`
  are automatically copied into the AAR's `jni/arm64-v8a/` directory by AGP.
  No extra packaging configuration is required.
- `.mnn` model assets in `MediapipeFaceLandmark/src/main/assets/` are merged
  into the AAR `assets/` and are accessible via the consumer's
  `assetManager.open("...")`.

### Multi-ABI

Only `arm64-v8a` is shipped today. To add `armeabi-v7a` / `x86_64`:
1. Rebuild MNN for the new ABI(s) and drop the resulting `.so` into
   `MediapipeFaceLandmark/src/main/jniLibs/<abi>/`.
2. Remove the `abiFilters 'arm64-v8a'` constraint in both
   `MediapipeFaceLandmark/build.gradle` and `facecapture-sdk/build.gradle`.

---

## 9. Versioning

`0.x.y` SemVer:
- **Major** — public Kotlin contract change in `com.facecapture.sdk.*`.
- **Minor** — additive features, new BlendShape outputs, smoothing modes.
- **Patch** — bug fixes, model retraining without API change.

The native ABI of `libmediapipefacelandmark.so` ↔ Kotlin JNI declarations
in `MediaPipeFaceDetector` must be considered part of the public ABI even
though they are technically internal classes — bumping MNN majorly or
changing native signatures is a **major** version bump.

---

## 10. License

Inherits the project root license. MNN is Apache-2.0. MediaPipe topology /
model files retain their original Apache-2.0 license from upstream Google.
OpenSeeFace post-processing logic was re-implemented from the original
BSD-2-Clause Python reference (see `OpenSeeFaceSource/LICENSE`).
