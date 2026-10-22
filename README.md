# CinnaGrade AI — research prototype

The current repository includes the clean-dataset ONNX model and SHA-bound deployment manifest. Extra Special log-bias 0.6 defaults ON for this model; original and corrected probabilities, paired evaluation, and research CSV exports are implemented. See [model deployment and validation](MODEL_DEPLOYMENT.md) for current behavior, checks, and limitations.

## Android release builds with Flutter 3.47.7

The Android build configuration uses Gradle **9.1.0**, Android Gradle Plugin (AGP) **9.0.1**, and Kotlin Gradle Plugin **2.3.20**, matching Flutter 3.47.7's recommended versions in its dependency validator. Use JDK **17 or 21**. The Gradle distribution is pinned with its official SHA-256 checksum.

AGP 9 changes both the Android DSL and Kotlin integration. The project uses `android.newDsl=false` and `android.builtInKotlin=false`, following the Flutter 3.47.7 app template's compatibility settings for Flutter/native plugins. The explicit Kotlin plugin remains enabled. Java and Kotlin bytecode targets both use 17; Kotlin uses the typed `compilerOptions` DSL instead of deprecated `kotlinOptions`. These switches retain supported plugin interfaces; they do not bypass Flutter's build dependency validation.

From PowerShell in your existing checkout:

```powershell
git pull --ff-only origin main
flutter clean
flutter pub get
flutter build apk --release
```

Allow the Gradle command to finish. The first build after changing toolchains may spend time downloading Gradle, Maven dependencies, SDK/NDK components, and compiling plugins. `Running Gradle task 'assembleRelease'...` and support/deprecation warnings alone do not establish a build failure. The supplied follow-up log ended with `Terminate batch job (Y/N)? y`, which indicates an interrupted build. A completed build reports either its APK output or a concrete failure.

A successful build writes `build\app\outputs\flutter-apk\app-release.apk`. `flutter pub get` may update local package resolutions to match your installed Flutter SDK. The Android SDK, platform/build tools, and any NDK requested by Flutter/plugins must be installed and their licenses accepted. Use `flutter doctor -v` to identify missing local prerequisites. Do not use `--android-skip-build-dependency-validation` as the fix.

Version requirements were checked against [Flutter 3.47.7's dependency validator](https://github.com/flutter/flutter/blob/3.47.7/packages/flutter_tools/gradle/src/main/kotlin/DependencyVersionChecker.kt), [compatibility tables](https://github.com/flutter/flutter/blob/3.47.7/packages/flutter_tools/lib/src/android/gradle_utils.dart), and the official [Gradle properties template](https://github.com/flutter/flutter/blob/3.47.7/packages/flutter_tools/templates/app/android.tmpl/gradle.properties.tmpl). This update replaces the previously supported 8.14.3 / 8.11.1 / 2.2.20 combination to address Flutter's advance support warnings.

Validation of this update: the Gradle 9.1.0 archive passed its official SHA-256 check and ran on JDK 21. An isolated Gradle task loaded the actual AGP 9.0.1 and Kotlin 2.3.20 plugins successfully. A second isolated Android application configuration, derived from the app Gradle script with Flutter values supplied as constants, successfully applied both plugins and the Java/Kotlin 17 DSL. ES-bias unit checks passed. This does not compile the actual Flutter application or validate an APK/device.

The compatibility switches above emit AGP/Kotlin deprecation warnings about a future migration; those warnings may remain and are not build failures. This update addresses the three version-support warnings in the supplied log without hiding warnings. The existing NDK 27.0.12077973 and minimum Android SDK 24 settings from your latest commit are retained.

A complete cloud APK build remains unverified: downloading the Flutter 3.47.7 Dart SDK was denied by the cloud proxy with HTTP 403, and a complete Android platform SDK is not installed. Run the commands above on your configured Windows machine and let the build complete.

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
