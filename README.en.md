# VirtualFaceCapture-MNN
> ✨ An Android virtual-avatar face-capture system, powered by the [alibaba-MNN](https://github.com/alibaba/MNN) inference engine.

> 🌐 中文版 README: [README.md](./README.md)

___

## 🧠 Overview
A mobile-side reimplementation of **MediaPipe Face Landmarker** on top of the
**MNN** inference engine, paired with an **OpenSeeFace-style** post-processing
pipeline to drive **Live2D** avatars in real time.
(After trying many alternatives it turns out Google's models are still the only
ones that hold up well on mobile -.-)

The whole capture pipeline has been completely **stripped out of the MediaPipe
source tree**: no `CalculatorGraph`, no JavaCV / OpenCV-Android, no MediaPipe
Tasks Java layer. Everything runs on **MNN + MNN-CV + native C++ post-processing**,
with hot paths (pose estimation, point-cloud transforms, …) accelerated by
**ARM NEON** SIMD intrinsics. End-to-end the system pushes close to **60 FPS**
on a Snapdragon 888 device. 🎉

---

## ✨ Demo

<p align="center">
  <img src="./sensi.gif" width="45%" />
  <img src="./zzzz.gif" width="45%" />
</p>

---

## 🧠 Architecture at a Glance

In one sentence:

> **Port MediaPipe Face Landmarker's Detection + Landmark models onto MNN,
> rebuild the pre/post image-processing in native code, then bolt on an
> OpenSeeFace-style smoothing & expression solver to emit ARKit-52 BlendShapes
> + head pose for driving Live2D.**

Differences vs. the official MediaPipe Android stack:

| Aspect | Official MediaPipe Tasks | This project (MNN port) |
|--------|--------------------------|--------------------------|
| Inference engine | TFLite + GPU Delegate | **MNN** (CPU / GPU / NNAPI backends) |
| Graph scheduling | MediaPipe `CalculatorGraph` | **Native C++ pipeline**, zero graph deps |
| Image processing | JavaCV / OpenCV / GL | **MNN-CV** + **NEON kernels** |
| Model count | Detection + Landmark + Blendshape | **Detection + Landmark only**, BlendShape solved geometrically |
| Post-processing | Built into MediaPipe | **OpenSeeFace-style** smoothing / calibration / head-pose recovery |
| Footprint | `mediapipe_tasks_vision.aar` | Just `libMNN.so` + `libmediapipefacelandmark.so` + 2 `.mnn` files |

---

## 🧩 Pipeline

```
            ┌───────────────────────────────┐
Camera ───► │ YUV_420_888 (Camera2/CameraX) │
            └──────────────┬────────────────┘
                           │  NEON YUV→RGB / rotation / resize
                           ▼
              ┌──────────────────────────┐
              │  FaceDetector  (MNN)     │   face_detector_mp.mnn
              │   - BlazeFace style      │   (int8-quantized variant included)
              └──────────────┬───────────┘
                             │ ROI affine crop (MNN-CV)
                             ▼
              ┌──────────────────────────┐
              │  FaceLandmark (MNN)      │   face_landmark_detector_mp.mnn
              │   - 478 3D landmarks     │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────────────────────┐
              │  OpenSeeFace-style post-processing (C++/Kt)  │
              │   • PnP / SVD head-pose recovery (p/y/r)     │
              │   • Blink curve + mouth + brows              │
              │   • One-Euro / rolling-mean smoothing        │
              │   • Auto baseline calibration (1 s window)   │
              │   • ARKit-52 BlendShape solver               │
              └──────────────┬───────────────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │  Live2D Cubism Renderer  │   ARKit → Live2D mapping
              └──────────────────────────┘
```

Linear algebra (matrix multiplies, affine transforms, SVD, Jacobi rotations,
3×N point-cloud transforms…) all goes through local `mini_linalg` +
`neon_intrinsics` headers, vectorized for the typical 3×N landmark shapes —
load time and first-frame latency drop noticeably vs. the scalar baseline.

---

## 📦 Modules

Standard multi-module Gradle project:

| Module | Purpose |
|--------|---------|
| `:app` | Demo app: camera, UI, Live2D rendering, parameter mapping |
| `:Live2D` | Cubism SDK wrapper (CMake + JNI) |
| `:MediapipeFaceLandmark` | **Core inference**: MNN + `.mnn` models + native C++ pipeline + NEON |
| `:OpenSeeFaceProcess` | OpenSeeFace-style post-processing: smoothing, head pose, BlendShape solver |
| `:CommonData` | Shared types (`Point3D` / `HeadPose` / `ARKitBlendShapes` …) |
| `:facecapture-sdk` | Public AAR facade wrapping the three modules above |
| `ThirdParty/MNN/` | MNN headers only |

---

## 🚀 Quick Start

### Build & run

```bash
git clone https://github.com/<your-name>/VirtualFaceCapture-MNN.git
cd VirtualFaceCapture-MNN
./gradlew :app:assembleDebug
# Or just open the project in Android Studio and hit Run.
```

Grant the camera permission on first launch, keep your face in frame, tap
**Start**, and the Live2D avatar will mirror your expression & head pose
in real time.

### Use as an SDK

If you only need the capture pipeline (no Live2D rendering), see
[`facecapture-sdk/README.md`](facecapture-sdk/README.md). The facade API
looks like:

```kotlin
val engine = FaceCaptureEngine.create()
engine.initialize(context)
engine.setResultListener { result ->
    // --- Raw perception layer (MediaPipe FaceLandmarker on MNN) ---
    val landmarks = result.landmarks         // 478 normalized 3D points
    val imgW      = result.imageWidth
    val imgH      = result.imageHeight
    val bbox      = result.detectionBox
    val presence  = result.presenceScore     // model confidence in [0, 1]

    // --- Post-processed layer (OpenSeeFace-style stack) ---
    val blendShapes = result.blendShapes     // 52 ARKit BlendShapes, smoothed
    val rawPose     = result.rawHeadPose     // raw pitch / yaw / roll
    val relPose     = result.relativeHeadPose // baseline-subtracted
}
// Push frames from CameraX:
engine.pushFrame(image, rotationDegrees)
```

Both layers (raw landmarks **and** post-processed BlendShape / head pose) come
from the same frame atomically — integrators can pick whichever they need.

---

## 🎮 Controls

When the demo launches, a scrollable **ControlPanel** appears at the bottom of
the screen with the current Live2D model name on top. All buttons and sliders:

### Buttons

| Button | Toggled state | Effect |
|--------|---------------|--------|
| **Start / Stop** | Red ↔ primary | Start / stop the whole capture pipeline (camera + inference + driving). When stopped, the avatar freezes at the last pose. |
| **Show Debug / Hide Debug** | — | Show or hide the on-screen debug HUD: FPS, inference timings, head pitch/yaw/roll, EAR, raw BlendShape values, etc. |
| **Show Camera / Show Live2D** | tertiary ↔ secondary | Switch the main view between the Live2D render and the camera preview (with landmarks overlay) — handy for diagnosing tracking issues. |
| **Select Model** | — | Open a dialog to pick a Live2D model from `assets/` (sensi / ATRI / … bundled). |
| **Reset Baseline** | — | Recalibrate: treat the current frame as the "neutral face" for head pose and expressions. Recommended whenever you put on glasses, change lighting, or hand the phone to a different person. |
| **Blink Curve: ON / OFF** | Green ↔ secondary | Toggle the **natural-blink curve shaper**. When on, raw blink events are reshaped into a smoother ~230 ms curve (no jitter / half-blinks). When off, the raw EAR is passed through. |

### Sliders

| Slider | Range | Notes |
|--------|-------|-------|
| **Blink Speed** *(only visible when Blink Curve is ON)* | 0.5x – 3.0x | Controls the duration of the reshaped blink. Larger = slower, smoother blink. The panel also shows the resulting duration in ms (e.g. `1.0x (230ms)`). `< 0.8x` is labeled "Fast blink", `> 1.5x` "Slow & smooth". |
| **Eye Sensitivity** | 0.5x – 2.0x | Sensitivity multiplier for the EAR thresholds. The panel shows the live `closed / open` thresholds below. `< 0.8x` → larger eye motion required to trigger (great for heavy makeup / glasses); `> 1.3x` → catches subtler motion (great for expressive recording). |

> 💡 TIP: On a fresh run, or after switching model / user, look straight at the
> camera once and tap **Reset Baseline**, then nudge **Eye Sensitivity** until
> blinks register reliably for your eye shape.

---

## 🛠 Highlights

- **Zero MediaPipe runtime dependency** — every `CalculatorGraph`,
  `PacketCallback`, `AndroidPacketCreator` etc. is gone; the model pipeline is
  scheduled by plain C++, with Kotlin running a multi-threaded parallel
  pipeline on top.
- **MNN-CV instead of OpenCV** — YUV decoding, resize, affine warp,
  normalization all live in MNN-CV. Saves several MB of `opencv-android`.
- **NEON SIMD** — `neon_intrinsics.h` implements dot products, matrix
  multiplies, 3×N outer products, Jacobi rotations, 4×4 affine transforms
  and friends; used by the SVD head-pose solver and the per-frame batch
  transform of 478 landmarks. Noticeable speed-up vs. scalar code.
- **OpenSeeFace-style post-processing** — rewrote bits of the Python
  `tracker.py` / `similaritytransform.py` in C++/Kotlin: One-Euro smoothing,
  first-frame calibration, blink curve shaping.
- **Geometric ARKit-52 solver** — does *not* require MediaPipe's third
  (BlendShape) model; solves the 52 BlendShapes from the 478 landmark
  geometry directly, saving one inference per frame. (Honestly the main
  reason is that the BlendShape model's raw output was hard to rescue;
  the inference code is still in the tree if you want to play with it :( .)
- **Publishable AAR** — `:facecapture-sdk` is already wired with
  `maven-publish` for one-shot release.

---

## 📁 Tree

```
VirtualFaceCapture-MNN/
├── app/                         # Demo + Live2D integration
├── Live2D/                      # Cubism SDK wrapper
├── MediapipeFaceLandmark/       # MNN inference + NEON-optimized core
│   ├── src/main/assets/         #   .mnn models (detector / landmark)
│   ├── src/main/cpp/            #   C++ pipeline, YUV, geometry
│   │   └── include/face_geomentry/
│   │       ├── mini_linalg.h        # linear algebra
│   │       └── neon_intrinsics.h    # ARM NEON SIMD
│   └── src/main/jniLibs/        #   prebuilt libMNN.so
├── OpenSeeFaceProcess/          # OpenSeeFace-style post-processing
├── CommonData/                  # Shared data types
├── facecapture-sdk/             # Public AAR facade
└── ThirdParty/MNN/              # MNN headers only
```

---

## 💡 Known Limitations

Because every face is shaped differently and the per-point mapping is hard to
nail down to clean 0–1 values, many thresholds in `OpenSeeFaceProcess` end up
as engineering trick — there is no single magic constant that fits every case.
If you have a better post-processing idea, PRs are very welcome (￣▽￣)"

---

## 🙏 Credits

- Live2D model **sensi**: <https://www.bilibili.com/video/BV1tk4y1n7Ls/>
- Live2D model **ATRI**: <https://www.bilibili.com/video/BV1Rs4y187rJ>
- [google/mediapipe](https://github.com/google/mediapipe)
- [emilianavt/OpenSeeFace](https://github.com/emilianavt/OpenSeeFace)
- [alibaba/MNN](https://github.com/alibaba/MNN)
