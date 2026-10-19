// Run with: dart --disable-dart-dev test/research_prediction_test.dart
// Dependency-free unit checks keep the application's package lock unchanged.
import 'dart:math' as math;
import '../lib/research_prediction.dart';

void check(bool condition, String message) {
  if (!condition) throw StateError(message);
}
void near(double actual, double expected) => check((actual - expected).abs() < 1e-10, '$actual != $expected');
void rejects(void Function() action) {
  var rejected = false;
  try { action(); } catch (_) { rejected = true; }
  check(rejected, 'Invalid values must be rejected');
}
Result sample({bool enabled = true}) => Result(
  rawProbs: [0.3, 0.4, 0.2, 0.1],
  correctedProbs: applyExtraSpecialBias([0.3, 0.4, 0.2, 0.1]),
  biasEnabled: enabled, bias: 0.6, modelVersion: 'test', modelSha256: 'test',
  runtimePlatform: 'test', inferenceProvider: 'test', preprocessMs: 2,
  inferenceMs: 4, postprocessingMs: 1, totalMs: 7,
  imageFingerprint: 'same-image', timestamp: DateTime.utc(2026, 10, 19));
void main() {
  check(kClasses.join(',') == 'extra_special,high_quality,low_quality,medium_quality', 'Class ordering changed');
  final raw = [0.3, 0.4, 0.2, 0.1];
  final corrected = applyExtraSpecialBias(raw);
  final denominator = 0.3 * math.exp(0.6) + 0.7;
  near(corrected[0], 0.3 * math.exp(0.6) / denominator);
  near(corrected[1], 0.4 / denominator);
  near(corrected[1] / corrected[2], 2);
  near(corrected.reduce((a, b) => a + b), 1);
  near(raw[0], 0.3);
  for (var i = 0; i < 4; i++) near(applyExtraSpecialBias(raw, bias: 0)[i], raw[i]);
  near(applyExtraSpecialBias([0, 1, 0, 0])[1], 1);
  rejects(() => applyExtraSpecialBias([1, 0]));
  rejects(() => applyExtraSpecialBias([double.nan, 0, 0, 1]));
  rejects(() => applyExtraSpecialBias([0.4, 0.4, 0.4, 0.4]));
  rejects(() => applyExtraSpecialBias(raw, bias: double.infinity));
  rejects(() => modelProbabilities([double.infinity, 1, 2, 3], outputsProbabilities: false));
  final stable = modelProbabilities([10000, 9999, 9998, 9997], outputsProbabilities: false);
  near(stable.reduce((a, b) => a + b), 1);
  for (var i = 0; i < 4; i++) near(modelProbabilities(raw, outputsProbabilities: true)[i], raw[i]);
  final enabled = sample(); final disabled = sample(enabled: false);
  check(enabled.rawLabel == 'high_quality', 'Original grade must be preserved');
  check(enabled.label == 'extra_special', 'Enabled correction must select corrected grade');
  check(enabled.withBiasEnabled(false).label == 'high_quality', 'Toggle must update final grade');
  check(disabled.label == 'high_quality', 'Disabled correction must select original grade');
  check(enabled.csvRow('a,"b.jpg').contains('"a,""b.jpg"'), 'CSV escaping failed');
  check(enabled.csvRow('x').split(',').length == Result.csvHeader.split(',').length, 'CSV header/row mismatch');
  check(enabled.csvRow('x').contains(',"","","",'), 'Unlabelled correctness must be empty');
  final evaluation = ResearchEvaluation();
  evaluation.record(enabled, 'extra_special');
  evaluation.record(enabled, 'extra_special');
  final report = evaluation.report();
  check(report.contains('1 unique expert-verified'), 'Repeated images must be deduplicated');
  check(report.contains('Original accuracy: 0.0%') && report.contains('Corrected accuracy: 100.0%'), 'Paired accuracy mismatch');
  print('PASS: class order, ES formula, normalization, edge cases, stable softmax, ON/OFF, CSV, paired unique-image metrics.');
}
