import 'dart:math' as math;

// This order is the ONNX classifier's output contract, not alphabetical order.
const List<String> kClasses = [
  'extra_special', 'high_quality', 'low_quality', 'medium_quality',
];
const double kGateThreshold = 0.52;
const double extraSpecialBias = 0.6;

void validateProbabilities(List<double> p) {
  if (p.length != 4 || p.any((v) => !v.isFinite || v < 0 || v > 1) ||
      (p.fold<double>(0, (a, b) => a + b) - 1).abs() > 1e-5) {
    throw StateError('Expected four finite, normalized class probabilities.');
  }
}

/// Reweight probabilities only after softmax; never apply softmax twice.
List<double> applyExtraSpecialBias(List<double> p, {double bias = extraSpecialBias}) {
  validateProbabilities(p);
  if (!bias.isFinite || bias.abs() > 20) {
    throw ArgumentError('ES log-bias must be finite and within [-20, 20].');
  }
  final adjusted = List<double>.from(p);
  adjusted[0] *= math.exp(bias);
  final total = adjusted.fold<double>(0, (a, b) => a + b);
  final corrected = adjusted.map((v) => v / total).toList();
  validateProbabilities(corrected);
  return corrected;
}

List<double> modelProbabilities(List<double> values, {required bool outputsProbabilities}) {
  if (values.length != 4 || values.any((v) => !v.isFinite)) {
    throw StateError('ONNX output must contain exactly four finite values.');
  }
  if (outputsProbabilities) {
    validateProbabilities(values);
    return List<double>.from(values);
  }
  final maximum = values.reduce(math.max);
  final exp = values.map((v) => math.exp(v - maximum)).toList();
  final total = exp.fold<double>(0, (a, b) => a + b);
  final p = exp.map((v) => v / total).toList();
  validateProbabilities(p);
  return p;
}

int topClass(List<double> p) {
  validateProbabilities(p);
  var best = 0;
  for (var i = 1; i < p.length; i++) {
    if (p[i] > p[best]) best = i;
  }
  return best;
}

class Result {
  final List<double> rawProbs;
  final List<double> correctedProbs;
  final bool biasEnabled;
  final double bias;
  final String modelVersion;
  final String modelSha256;
  final String runtimePlatform;
  final String inferenceProvider;
  final int preprocessMs;
  final int inferenceMs;
  final int postprocessingMs;
  final int totalMs;
  final String imageFingerprint;
  final DateTime timestamp;

  Result({required List<double> rawProbs, required List<double> correctedProbs,
    required this.biasEnabled, required this.bias, required this.modelVersion,
    required this.modelSha256, required this.runtimePlatform,
    required this.inferenceProvider, required this.preprocessMs,
    required this.inferenceMs, required this.postprocessingMs,
    required this.totalMs, required this.imageFingerprint, DateTime? timestamp})
    : rawProbs = List.unmodifiable(rawProbs),
      correctedProbs = List.unmodifiable(correctedProbs),
      timestamp = timestamp ?? DateTime.now().toUtc();

  String get rawLabel => kClasses[topClass(rawProbs)];
  String get correctedLabel => kClasses[topClass(correctedProbs)];
  double get rawConfidence => rawProbs[topClass(rawProbs)];
  double get correctedConfidence => correctedProbs[topClass(correctedProbs)];
  List<double> get probs => biasEnabled ? correctedProbs : rawProbs;
  String get label => biasEnabled ? correctedLabel : rawLabel;
  double get confidence => biasEnabled ? correctedConfidence : rawConfidence;
  bool get accepted => confidence >= kGateThreshold;

  Result withBiasEnabled(bool enabled) => Result(
    rawProbs: rawProbs, correctedProbs: correctedProbs, biasEnabled: enabled,
    bias: bias, modelVersion: modelVersion, modelSha256: modelSha256,
    runtimePlatform: runtimePlatform, inferenceProvider: inferenceProvider,
    preprocessMs: preprocessMs, inferenceMs: inferenceMs,
    postprocessingMs: postprocessingMs, totalMs: totalMs,
    imageFingerprint: imageFingerprint, timestamp: timestamp,
  );

  static const csvHeader = 'timestamp_utc,image_filename,image_fingerprint,model_version,model_sha256,extra_special_bias,bias_enabled,raw_predicted_grade,raw_confidence,corrected_predicted_grade,corrected_confidence,final_predicted_grade,raw_prob_extra,raw_prob_high,raw_prob_low,raw_prob_medium,corrected_prob_extra,corrected_prob_high,corrected_prob_low,corrected_prob_medium,expert_grade,raw_is_correct,corrected_is_correct,preprocessing_ms,inference_ms,postprocessing_ms,total_latency_ms,runtime_platform,inference_provider,model_load_status';

  String csvRow(String filename, {String expertGrade = ''}) {
    if (expertGrade.isNotEmpty && !kClasses.contains(expertGrade)) {
      throw ArgumentError('Unknown expert grade.');
    }
    // Empty correctness cells preserve the distinction between unlabelled and wrong.
    final values = <Object>[
      timestamp.toUtc().toIso8601String(), filename, imageFingerprint,
      modelVersion, modelSha256, bias, biasEnabled, rawLabel, rawConfidence,
      correctedLabel, correctedConfidence, label, ...rawProbs, ...correctedProbs,
      expertGrade, expertGrade.isEmpty ? '' : (expertGrade == rawLabel ? 1 : 0),
      expertGrade.isEmpty ? '' : (expertGrade == correctedLabel ? 1 : 0),
      preprocessMs, inferenceMs, postprocessingMs, totalMs, runtimePlatform,
      inferenceProvider, 'loaded',
    ];
    return values.map((v) => '"${v.toString().replaceAll('"', '""')}"').join(',');
  }
}

/// Report paired raw/corrected metrics on the same unique, expert-labelled images.
class ResearchEvaluation {
  final Map<String, (Result, String)> _images = {};
  void record(Result result, String expertGrade) {
    if (kClasses.contains(expertGrade)) {
      _images[result.imageFingerprint] = (result, expertGrade);
    }
  }

  String report() {
    if (_images.isEmpty) return 'No expert-verified images yet.';
    final lines = <String>['${_images.length} unique expert-verified images (latest label per fingerprint).'];
    for (final corrected in [false, true]) {
      final matrix = List.generate(4, (_) => List.filled(4, 0));
      for (final entry in _images.values) {
        matrix[kClasses.indexOf(entry.$2)][kClasses.indexOf(
          corrected ? entry.$1.correctedLabel : entry.$1.rawLabel)]++;
      }
      var correct = 0;
      var macroF1 = 0.0;
      final scores = <String>[];
      for (var i = 0; i < 4; i++) {
        final tp = matrix[i][i];
        correct += tp;
        final support = matrix[i].reduce((a, b) => a + b);
        final predicted = matrix.fold<int>(0, (a, row) => a + row[i]);
        final precision = predicted == 0 ? 0.0 : tp / predicted;
        final recall = support == 0 ? 0.0 : tp / support;
        final f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
        macroF1 += f1 / 4;
        scores.add('${kClasses[i]} (n=$support): P=${precision.toStringAsFixed(3)}, R=${recall.toStringAsFixed(3)}, F1=${f1.toStringAsFixed(3)}');
      }
      lines.add('${corrected ? 'Corrected' : 'Original'} accuracy: ${(100 * correct / _images.length).toStringAsFixed(1)}% ($correct/${_images.length})');
      lines.addAll(scores);
      lines.add('Macro F1 (all four classes): ${macroF1.toStringAsFixed(3)}');
      lines.add('Confusion matrix: rows=expert, columns=prediction; class order: ${kClasses.join(', ')}');
      lines.addAll(matrix.map((row) => row.join(', ')));
    }
    lines.add('Descriptive sample results, not guaranteed field accuracy. Undefined precision/recall use 0; absent classes have support 0.');
    return lines.join('\n');
  }
}
