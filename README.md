# CinnaGrade AI — research prototype

The current repository includes the clean-dataset ONNX model and SHA-bound deployment manifest. Extra Special log-bias 0.6 defaults ON for this model; original and corrected probabilities, paired evaluation, and research CSV exports are implemented. See [model deployment and validation](MODEL_DEPLOYMENT.md) for current behavior, checks, and limitations.

## Android release builds with Flutter 3.47.7

The Android build configuration uses Gradle **8.14.3**, Android Gradle Plugin (AGP) **8.11.1**, and Kotlin Gradle Plugin **2.2.20**. These meet Flutter 3.47.7's minimum supported versions (8.14.0 / 8.11.1 / 2.2.20). Use JDK **17 or 21** for this build configuration. The Gradle distribution is pinned with its official SHA-256 checksum.

From PowerShell in your existing checkout:

```powershell
git pull --ff-only origin main
flutter clean
flutter pub get
flutter build apk --release
```

A successful build writes `build\app\outputs\flutter-apk\app-release.apk`. `flutter pub get` may update local package resolutions to match your installed Flutter SDK. The Android SDK, platform/build tools, and any NDK requested by Flutter/plugins must be installed and their licenses accepted. Use `flutter doctor -v` to identify missing local prerequisites.

The failure saying Gradle 8.12 is below Flutter's supported minimum is fixed by the checked-in wrapper properties; reinstalling Flutter or running `flutter pub add --dev flutter_launcher_icons` does not fix it. Do not use `--android-skip-build-dependency-validation` as the fix. The additional AGP 9/new DSL hint in the supplied log is generic: this project remains on AGP 8.11.1, so it does not require an AGP 9 migration or `android.newDsl` opt-out.

Validation of this update: the Gradle 8.14.3 distribution passed SHA-256 verification and ran on JDK 21; an isolated Gradle task resolved and loaded the actual AGP 8.11.1 and Kotlin 2.2.20 plugins successfully. The existing ES-bias unit checks also passed. This checks toolchain resolution, not compilation of the Android application. Flutter may still show advance warnings recommending newer major versions; the selected versions meet its hard minimums while keeping the AGP 8 DSL.

Version requirements were checked against [Flutter 3.47.7's dependency validator](https://github.com/flutter/flutter/blob/3.47.7/packages/flutter_tools/gradle/src/main/kotlin/DependencyVersionChecker.kt) and [its Gradle/Kotlin/AGP compatibility tables](https://github.com/flutter/flutter/blob/3.47.7/packages/flutter_tools/lib/src/android/gradle_utils.dart). A cloud configuration check is not an Android APK/device validation; this cloud machine currently has no Android SDK.

The historical setup notes below refer to the earlier source ZIP. Their missing-model warning and temperature calibration do not describe the current bundled deployment.

# Cinnamon Edge AI — Flutter prototype

**Important:** This ZIP contains Flutter source code; it intentionally does NOT contain the user's private 20.34 MB ONNX model. It has not been built on a local Android device in this environment. Dart/plugin compatibility and ONNX I/O must be checked on the user's system before a working-device claim.

## Setup on Windows

1. Install Flutter & Android Studio. Open Windows PowerShell.
2. `flutter create cinnamon_edge` (or `flutter create . --platforms android` if applying inside the project directory).
3. Copy this ZIP's `lib/main.dart`, `pubspec.yaml`, and `assets/` into the generated Flutter project. Preserve generated android/gradle files.
4. From Google Drive Experiment21 `20261015_162548`, download `frozen_yolov8s_fp32_384.onnx` and place it at `assets/models/frozen_yolov8s_fp32_384.onnx`.
5. Confirm ONNX input name is `images` (if not, change `inputName` near top of main.dart); output must be a batch of 4 logits or 4 probabilities in order `extra_special, high_quality, low_quality, medium_quality`.
6. Add `android/app/proguard-rules.pro` text: `-keep class ai.onnxruntime.** { *; }`. Ensure Android project uses camera permission where required.
7. `flutter pub get` then `flutter run` on Android. Flutter version may require package upgrades; latest image_picker release requires Flutter >=3.41, so older compatible ^1.1.2 is pinned here.
8. Select Camera/Gallery. After a prediction choose optional expert-confirmed label and Save; Share CSV for research.

## Caveats

- Captured photos are processed with letterbox 384px RGB padding 128, CHW normalized 0..1, EXIF orientation baked. Compare image tensor and probabilities with Python reference before trusting labels.
- Calibrated temperature=1.457365 and threshold=0.52 came from reused validation and are NOT field-proven.
- CSV keeps predicted labels and times for each saved image, but accuracy is computed ONLY from explicitly expert-verified entries. No pseudo-accuracy is fabricated.
- Timings are wall-clock app preprocessing and ONNX session run; not phone power, not startup time; record phone model separately.
- Prediction log rows are written only when tapping Save; do not repeatedly tap Save for the same image.
- ONNX Runtime Flutter package requires Android-specific setup documented at https://pub.dev/packages/flutter_onnxruntime .
