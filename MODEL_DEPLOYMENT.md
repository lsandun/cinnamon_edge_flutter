# CinnaGrade AI research deployment

The bundled ONNX graph is the clean-data model exported on 2026-10-19. Its metadata declares the class order `extra_special, high_quality, low_quality, medium_quality`; its input is float32 `images` `[1,3,384,384]` and output is `output0` `[1,4]`. The graph ends in Softmax. `assets/models/deployment_manifest.json` binds the model SHA-256 to these deployment settings.

The app preserves RGB, EXIF orientation, aspect-ratio-preserving cubic resize, a 384-pixel canvas with RGB 128 padding, [0,1] normalization, and NCHW layout. No second Softmax or previous experiment's temperature is applied to this model's probabilities. Cross-language preprocessing equivalence still requires research reference fixtures; none were supplied.

Extra Special correction multiplies probability index 0 by `exp(0.6)` and renormalizes all four classes. This is the requested validation-selected log-bias, not an embedded model change or an accuracy guarantee. It defaults ON for this exact clean model. The switch updates the current single-image final grade and subsequent inference; saved/batch results retain their inference setting. Original and corrected probabilities remain available in either mode. Deploying another model requires a matching, verified manifest; never copy this model's calibration to an older model automatically.

Both single-image and batch exports share the 30-column research CSV schema. Rows retain full image SHA-256 fingerprints, model identity, correction setting, original/corrected/final predictions, expert labels, real Stopwatch timings, runtime, provider, and load status. Raw inference time includes output retrieval through the plugin; total time includes input preparation and post-processing, but excludes model initialization and UI animation. Millisecond rounding can legitimately produce zero for fast post-processing.

Repeated taps with identical prediction/setting/expert label are suppressed. Full exports retain separate intentional prediction records. Evaluation uses only expert-labelled images, deduplicates by fingerprint using the latest saved label/result, and reports paired original/corrected accuracy, class precision/recall/F1, macro F1, and confusion matrices. Rows are expert classes and columns are predictions. Undefined precision/recall use zero and support is shown; small samples are descriptive, not field validation.

## Validation commands

```sh
flutter pub get --enforce-lockfile
flutter analyze --no-pub
dart --disable-dart-dev test/research_prediction_test.dart
flutter build web --no-pub --no-web-resources-cdn
flutter build apk --release --no-pub
```

The dependency-free Dart test exits nonzero on failures and covers class ordering, ES reweighting, probability normalization, invalid inputs, stable Softmax, ON/OFF grade selection, CSV consistency/escaping, and paired unique-image evaluation. Dependency declarations and lockfile are unchanged.

Web uses the existing ONNX Runtime Web CDN script and therefore requires network access for that runtime; only Android inference is intended to be fully offline. The web build command bundles Flutter renderer resources locally. Android uses the CPU provider, with no new Google Play Services dependency. Flutter 3.32.0 configures minimum Android SDK 21, below the Huawei Y7a's Android 10 (API 29); actual APK/device testing is still required.

The expected Android output after a successful release build is `build/app/outputs/flutter-apk/app-release.apk`. No APK was produced in this cloud machine because its Android SDK is absent. No device accuracy or phone latency claim is made.

## Validation performed in this cloud machine

- Frozen-lockfile dependency installation passed; static analysis reported no issues.
- The Dart unit checks passed. No package upgrade or lockfile change was needed.
- Flutter Web release build passed with locally bundled renderer resources. The build emitted a Cupertino icon-font warning; the application still compiled successfully.
- Chromium initialized the real model, selected an existing repository PNG through Gallery, and displayed original/corrected confidences and all four paired probabilities with correction ON. The test inspected Flutter accessibility labels as well as page text. CDN runtime artifacts were fetched through the system TLS-verifying client during this browser check because Chromium does not trust the cloud proxy CA; the trust store was not changed. Cloud font-network failures limit visual verification. This is a technical smoke check using an app icon, not a cinnamon accuracy evaluation.
- Python ONNX schema checking, native CPU initialization, and inference on two distinct repository PNGs passed, with finite normalized four-class outputs that differed between inputs. No numerical Flutter/Python parity claim is made without shared research fixtures.
- Android release build was attempted and stopped because no Android SDK is installed. No APK or Huawei device test result is available.

Pre-change application files and package manifests were backed up outside the checkout at `/workspace/backups/cinnamon-edge-before-es-bias/`.
