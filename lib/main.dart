import 'dart:io' show File;
import 'dart:math' as math;
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'widgets/scan_animation.dart';

const List<String> kClasses = [
  'extra_special',
  'high_quality',
  'low_quality',
  'medium_quality',
];

const Map<String, String> kClassDisplayNames = {
  'extra_special': 'Extra Special (Alba / C5)',
  'high_quality': 'High Quality (C5 / M5)',
  'medium_quality': 'Medium Quality (M4 / H1)',
  'low_quality': 'Low Quality (H2 / Quillings)',
};

const double kTemperature = 1.4573651344965806; // Exp21 provisional calibration
const double kGateThreshold = 0.52; // Confidence acceptance gate
const String kInputName = 'images';
const String kModelAssetPath = 'assets/models/frozen_yolov8s_fp32_384.onnx';

class Result {
  final String label;
  final double confidence;
  final List<double> probs;
  final int preprocessMs;
  final int inferenceMs;
  final int totalMs;
  final bool accepted;
  final String imageFingerprint;

  Result({
    required this.label,
    required this.confidence,
    required this.probs,
    required this.preprocessMs,
    required this.inferenceMs,
    required this.totalMs,
    required this.accepted,
    required this.imageFingerprint,
  });
}

class EdgeEngine {
  OnnxRuntime? _runtimeInstance;
  OnnxRuntime get _runtime => _runtimeInstance ??= OnnxRuntime();
  OrtSession? _session;
  bool _modelAssetChecked = false;
  bool _modelAssetExists = false;

  bool get isLoaded => _session != null;

  Future<bool> checkModelExists() async {
    if (_modelAssetChecked) return _modelAssetExists;
    try {
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      _modelAssetExists = manifest.listAssets().contains(kModelAssetPath);
    } catch (_) {
      try {
        await rootBundle.load(kModelAssetPath);
        _modelAssetExists = true;
      } catch (_) {
        _modelAssetExists = false;
      }
    }
    _modelAssetChecked = true;
    return _modelAssetExists;
  }

  Future<void> load() async {
    if (_session != null) return;
    final exists = await checkModelExists();
    if (!exists) {
      throw Exception(
        'ONNX Model not found in assets ($kModelAssetPath). '
        'Please ensure "frozen_yolov8s_fp32_384.onnx" is present in "assets/models/".',
      );
    }
    _session = await _runtime.createSessionFromAsset(kModelAssetPath);
  }

  Float32List _tensor(img.Image source) {
    final oriented = img.bakeOrientation(source);
    final ratio = math.min(384 / oriented.width, 384 / oriented.height);
    final w = math.max(1, (oriented.width * ratio).round());
    final h = math.max(1, (oriented.height * ratio).round());
    final resized = img.copyResize(
      oriented,
      width: w,
      height: h,
      interpolation: img.Interpolation.cubic,
    );
    final canvas = img.Image(width: 384, height: 384);
    img.fill(canvas, color: img.ColorRgb8(128, 128, 128));
    img.compositeImage(
      canvas,
      resized,
      dstX: (384 - w) ~/ 2,
      dstY: (384 - h) ~/ 2,
    );

    final out = Float32List(3 * 384 * 384);
    for (var y = 0; y < 384; y++) {
      for (var x = 0; x < 384; x++) {
        final p = canvas.getPixel(x, y);
        final idx = y * 384 + x;
        out[idx] = p.r / 255.0;
        out[384 * 384 + idx] = p.g / 255.0;
        out[2 * 384 * 384 + idx] = p.b / 255.0;
      }
    }
    return out;
  }

  Future<Result> predict(
    String path, {
    Uint8List? imageBytes,
  }) async {
    if (_session == null) {
      await load();
    }
    if (_session == null) {
      throw StateError(
        'ONNX Session is not initialized. Model must be loaded before running inference.',
      );
    }

    final totalTimer = Stopwatch()..start();
    final prepTimer = Stopwatch()..start();

    final Uint8List bytes;
    if (imageBytes != null && imageBytes.isNotEmpty) {
      bytes = imageBytes;
    } else if (!kIsWeb) {
      bytes = await File(path).readAsBytes();
    } else {
      throw StateError('Image bytes required for Web inference.');
    }

    // Compute unique 16-char SHA-256 fingerprint for accurate image tracking
    final fingerprint = sha256.convert(bytes).toString().substring(0, 16);

    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Image could not be decoded. Ensure it is a valid JPEG/PNG.');
    }

    final nchw = _tensor(decoded);
    final prepMs = prepTimer.elapsedMilliseconds;

    final ortInput = await OrtValue.fromList(nchw.toList(), [1, 3, 384, 384]);
    final inferTimer = Stopwatch()..start();

    try {
      final output = await _session!.run({kInputName: ortInput});
      if (output.values.isEmpty) {
        throw StateError('ONNX model returned empty output tensor.');
      }
      final raw = await output.values.first.asList();
      final values = _flatten(raw);
      if (values.length != 4) {
        throw StateError(
          'Unexpected ONNX output shape: received ${values.length} values (expected 4).',
        );
      }

      // Check if output is raw logits or probabilities
      final alreadyProbs = values.every((v) => v >= 0 && v <= 1) &&
          (values.reduce((a, b) => a + b) - 1.0).abs() < 0.02;
      final logits = alreadyProbs
          ? values.map((v) => math.log(math.max(v, 1e-12))).toList()
          : values;

      final maxLogit = logits.reduce(math.max);
      final expv = logits.map((v) => math.exp((v - maxLogit) / kTemperature)).toList();
      final sumExp = expv.reduce((a, b) => a + b);
      final probabilities = expv.map((v) => v / sumExp).toList();

      var topIndex = 0;
      for (var i = 1; i < 4; i++) {
        if (probabilities[i] > probabilities[topIndex]) {
          topIndex = i;
        }
      }

      final inferMs = inferTimer.elapsedMilliseconds;
      final totalMs = totalTimer.elapsedMilliseconds;
      final confidence = probabilities[topIndex];

      return Result(
        label: kClasses[topIndex],
        confidence: confidence,
        probs: probabilities,
        preprocessMs: prepMs,
        inferenceMs: inferMs,
        totalMs: totalMs,
        accepted: confidence >= kGateThreshold,
        imageFingerprint: fingerprint,
      );
    } finally {
      ortInput.dispose();
    }
  }

  List<double> _flatten(dynamic x) {
    if (x is num) return [x.toDouble()];
    if (x is Iterable) return x.expand((v) => _flatten(v)).toList();
    throw StateError('Unknown tensor result type: ${x.runtimeType}');
  }
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CinnamonApp());
}

class CinnamonApp extends StatelessWidget {
  const CinnamonApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cinnamon Edge AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B4D3E),
          primary: const Color(0xFF1B4D3E),
          secondary: const Color(0xFFB3541E),
          surface: const Color(0xFFF9FAF8),
        ),
        cardTheme: const CardThemeData(
          elevation: 1,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
        ),
      ),
      home: const CinnamonHome(),
    );
  }
}

class CinnamonHome extends StatefulWidget {
  const CinnamonHome({super.key});

  @override
  State<CinnamonHome> createState() => _CinnamonHomeState();
}

class _CinnamonHomeState extends State<CinnamonHome> {
  final ImagePicker _picker = ImagePicker();
  final EdgeEngine _engine = EdgeEngine();

  String? _imagePath;
  Uint8List? _imageBytes;
  String? _imageName;
  String? _errorMessage;
  Result? _result;
  bool _isProcessing = false;
  String _scanStatusText = 'Reading cinnamon bark sample...';
  double _scanProgress = 0.2;

  bool _isModelLoading = true;
  bool _isModelLoaded = false;
  String? _modelLoadError;

  String _expertGrade = '';
  int _evaluatedCount = 0;
  int _correctCount = 0;
  int _savedRowCount = 0;
  String? _csvFilePath;

  final List<String> _csvRows = [
    'timestamp_utc,image_fingerprint,image,runtime_mode,grade,confidence,accepted,preprocess_ms,inference_ms,total_ms,expert_grade,is_correct,prob_extra,prob_high,prob_low,prob_medium'
  ];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
      _modelLoadError = null;
    });

    try {
      final exists = await _engine.checkModelExists();
      if (!exists) {
        throw Exception(
          'ONNX model file not found in assets/models/. Please ensure "frozen_yolov8s_fp32_384.onnx" is bundled.',
        );
      }
      await _engine.load();
      if (mounted) {
        setState(() {
          _isModelLoaded = true;
          _isModelLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isModelLoaded = false;
          _isModelLoading = false;
          _modelLoadError = e.toString();
        });
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_isModelLoaded) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('ONNX Model is not loaded. Cannot run inference.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final XFile? file = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 95,
      );
      if (file == null) return;

      final bytes = await file.readAsBytes();
      final name = file.name.isNotEmpty
          ? file.name
          : file.path.split(RegExp(r'[/\\]')).last;

      setState(() {
        _isProcessing = true;
        _errorMessage = null;
        _result = null;
        _imagePath = file.path;
        _imageBytes = bytes;
        _imageName = name;
        _expertGrade = '';
        _scanStatusText = 'Letterboxing 384×384 (NCHW) & normalizing...';
        _scanProgress = 0.35;
      });

      Future.delayed(const Duration(milliseconds: 300), () {
        if (mounted && _isProcessing) {
          setState(() {
            _scanStatusText = 'Executing YOLOv8s Neural Tensor Inference...';
            _scanProgress = 0.70;
          });
        }
      });

      Future.delayed(const Duration(milliseconds: 650), () {
        if (mounted && _isProcessing) {
          setState(() {
            _scanStatusText = 'Calibrating Temperature (T=1.457) & Gate...';
            _scanProgress = 0.95;
          });
        }
      });

      // Run REAL ONNX inference with minimum display duration (~800ms) for smooth UX
      final results = await Future.wait([
        _engine.predict(
          file.path,
          imageBytes: bytes,
        ),
        Future.delayed(const Duration(milliseconds: 800)),
      ]);

      final res = results[0] as Result;

      if (mounted) {
        setState(() {
          _scanProgress = 1.0;
          _result = res;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _result = null;
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<void> _savePrediction() async {
    if (_result == null || (_imagePath == null && _imageBytes == null)) return;
    final r = _result!;
    final hasExpertVerification = _expertGrade.isNotEmpty;
    final isCorrect = hasExpertVerification ? (_expertGrade == r.label ? 1 : 0) : null;

    if (hasExpertVerification) {
      _evaluatedCount++;
      if (isCorrect == 1) _correctCount++;
    }

    String esc(String s) => '"${s.replaceAll('"', '""')}"';
    final filename = _imageName ??
        (_imagePath != null
            ? _imagePath!.split(RegExp(r'[/\\]')).last
            : 'image.png');

    _csvRows.add([
      esc(DateTime.now().toUtc().toIso8601String()),
      esc(r.imageFingerprint),
      esc(filename),
      'REAL_ONNX_RUNTIME',
      esc(r.label),
      r.confidence.toStringAsFixed(6),
      r.accepted,
      r.preprocessMs.toString(),
      r.inferenceMs.toString(),
      r.totalMs.toString(),
      esc(_expertGrade),
      isCorrect?.toString() ?? '',
      ...r.probs.map((v) => v.toStringAsFixed(6)),
    ].join(','));

    _savedRowCount++;

    if (kIsWeb) {
      _csvFilePath = 'cinnamon_edge_predictions.csv';
      if (mounted) {
        setState(() {});
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Saved: $filename [${r.imageFingerprint}] ($_savedRowCount total logs)'),
            backgroundColor: const Color(0xFF1B4D3E),
            duration: const Duration(seconds: 2),
          ),
        );
      }
      return;
    }

    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/cinnamon_edge_predictions.csv');
    await file.writeAsString('${_csvRows.join('\n')}\n');
    _csvFilePath = file.path;

    if (mounted) {
      setState(() {});
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Saved: $filename [${r.imageFingerprint}] ($_savedRowCount total logs)'),
          backgroundColor: const Color(0xFF1B4D3E),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  Future<void> _shareCsv() async {
    if (_csvFilePath == null && _csvRows.length <= 1) return;
    if (kIsWeb) {
      await Share.share(
        _csvRows.join('\n'),
        subject: 'Cinnamon Edge Research Predictions (Field Trial)',
      );
      return;
    }
    await Share.shareXFiles(
      [XFile(_csvFilePath!)],
      text: 'Cinnamon Edge Research Predictions (Field Trial)',
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF7F9F6),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFF1B4D3E).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.eco, color: Color(0xFF1B4D3E), size: 22),
            ),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Cinnamon Edge AI',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF112211),
                  ),
                ),
                Text(
                  'Offline Research Prototype',
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey.shade600,
                  ),
                ),
              ],
            ),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 14),
            child: Center(
              child: _isModelLoading
                  ? Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFFF3E0),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: const Color(0xFFFF9800)),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 10,
                            height: 10,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFFE65100)),
                            ),
                          ),
                          SizedBox(width: 6),
                          Text(
                            'LOADING ONNX...',
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFFE65100),
                            ),
                          ),
                        ],
                      ),
                    )
                  : _isModelLoaded
                      ? Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: const Color(0xFFE8F5E9),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: const Color(0xFF4CAF50)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 8,
                                height: 8,
                                decoration: const BoxDecoration(
                                  color: Color(0xFF2E7D32),
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 6),
                              const Text(
                                'OFFLINE READY (REAL ONNX)',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF2E7D32),
                                ),
                              ),
                            ],
                          ),
                        )
                      : InkWell(
                          onTap: _loadModel,
                          borderRadius: BorderRadius.circular(16),
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: const Color(0xFFFFEBEE),
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: const Color(0xFFEF5350)),
                            ),
                            child: const Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.refresh, size: 12, color: Color(0xFFC62828)),
                                SizedBox(width: 4),
                                Text(
                                  'MODEL ERROR (RETRY)',
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold,
                                    color: Color(0xFFC62828),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        children: [
          // Model loading / error banner
          if (_isModelLoading) ...[
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFA5D6A7)),
              ),
              child: const Row(
                children: [
                  SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF1B4D3E)),
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Loading ONNX Neural Model (assets/models/frozen_yolov8s_fp32_384.onnx)...',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF1B4D3E),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
          ] else if (_modelLoadError != null) ...[
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFFFFEBEE),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFFFCDD2)),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(Icons.error_outline, color: Colors.red, size: 22),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'ONNX Model Initialization Failed',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                            color: Color(0xFFC62828),
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          _modelLoadError!,
                          style: const TextStyle(fontSize: 11.5, color: Color(0xFFB71C1C)),
                        ),
                        const SizedBox(height: 8),
                        ElevatedButton.icon(
                          onPressed: _loadModel,
                          icon: const Icon(Icons.refresh, size: 16),
                          label: const Text('Retry Loading Model'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFFC62828),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
          ],

          // Action Buttons: Camera & Gallery (Active ONLY when ONNX model is loaded)
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: (!_isModelLoaded || _isProcessing) ? null : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt, color: Colors.white),
                  label: Text(
                    _isModelLoading ? 'Loading Model...' : 'Take Photo',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
                      color: Colors.white,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1B4D3E),
                    disabledBackgroundColor: Colors.grey.shade400,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 2,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: (!_isModelLoaded || _isProcessing) ? null : () => _pickImage(ImageSource.gallery),
                  icon: Icon(
                    Icons.photo_library,
                    color: (!_isModelLoaded || _isProcessing) ? Colors.grey : const Color(0xFF1B4D3E),
                  ),
                  label: Text(
                    'Gallery',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
                      color: (!_isModelLoaded || _isProcessing) ? Colors.grey : const Color(0xFF1B4D3E),
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    side: BorderSide(
                      color: (!_isModelLoaded || _isProcessing) ? Colors.grey.shade300 : const Color(0xFF1B4D3E),
                      width: 1.5,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),

          // Error display
          if (_errorMessage != null) ...[
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFFFFEBEE),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFFFCDD2)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.red),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      _errorMessage!,
                      style: const TextStyle(fontSize: 12, color: Colors.red),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
          ],

          // Image Preview Container with AI Laser Scanner Overlay
          if (_imageBytes != null) ...[
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                color: Colors.black,
                boxShadow: [
                  BoxShadow(
                    color: _isProcessing
                        ? const Color(0xFF00E676).withValues(alpha: 0.18)
                        : Colors.black.withValues(alpha: 0.08),
                    blurRadius: _isProcessing ? 16 : 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Stack(
                  alignment: Alignment.bottomLeft,
                  children: [
                    Image.memory(
                      _imageBytes!,
                      height: 270,
                      width: double.infinity,
                      fit: BoxFit.contain,
                    ),

                    // High-tech laser scanner overlay while processing
                    if (_isProcessing)
                      Positioned.fill(
                        child: AiLaserScanner(
                          statusText: _scanStatusText,
                          progressPercent: _scanProgress,
                        ),
                      )
                    else
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.bottomCenter,
                            end: Alignment.topCenter,
                            colors: [
                              Colors.black.withValues(alpha: 0.85),
                              Colors.transparent,
                            ],
                          ),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Text(
                                _imageName ?? 'image.png',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 11.5,
                                  fontWeight: FontWeight.w500,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            if (_result != null)
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                                decoration: BoxDecoration(
                                  color: Colors.white.withValues(alpha: 0.2),
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  'FP: ${_result!.imageFingerprint}',
                                  style: const TextStyle(
                                    color: Color(0xFF00E676),
                                    fontSize: 10,
                                    fontFamily: 'monospace',
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
          ] else ...[
            // Empty State
            Container(
              padding: const EdgeInsets.symmetric(vertical: 36, horizontal: 20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade200),
              ),
              child: Column(
                children: [
                  Icon(Icons.camera_enhance_outlined, size: 48, color: Colors.grey.shade400),
                  const SizedBox(height: 10),
                  const Text(
                    'No Image Selected',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Capture a clear, well-lit photo of cinnamon bark or choose from gallery.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey.shade600, fontSize: 12.5),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
          ],

          // Shimmer Skeleton Card during processing (engaging UX feedback)
          if (_isProcessing) ...[
            const ShimmerResultCard(),
            const SizedBox(height: 14),
          ] else if (_result != null) ...[
            // Prediction Results Card
            Card(
              color: Colors.white,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: const Color(0xFFE8F5E9),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: const Color(0xFFA5D6A7)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.verified, size: 13, color: Color(0xFF2E7D32)),
                          const SizedBox(width: 5),
                          Text(
                            'REAL_ONNX_RUNTIME (Fingerprint: ${_result!.imageFingerprint})',
                            style: const TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF2E7D32),
                              fontFamily: 'monospace',
                            ),
                          ),
                        ],
                      ),
                    ),

                    // Predicted Grade & Confidence Badge
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'PREDICTED GRADE',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  letterSpacing: 0.8,
                                  color: Colors.grey.shade600,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                kClassDisplayNames[_result!.label] ?? _result!.label,
                                style: const TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF1B4D3E),
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: _result!.accepted
                                ? const Color(0xFFE8F5E9)
                                : const Color(0xFFFFEBEE),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: _result!.accepted
                                  ? const Color(0xFF81C784)
                                  : const Color(0xFFE57373),
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.end,
                            children: [
                              Text(
                                '${(_result!.confidence * 100).toStringAsFixed(1)}%',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  color: _result!.accepted
                                      ? const Color(0xFF2E7D32)
                                      : const Color(0xFFC62828),
                                ),
                              ),
                              Text(
                                'Confidence',
                                style: TextStyle(
                                  fontSize: 10,
                                  color: _result!.accepted
                                      ? const Color(0xFF2E7D32)
                                      : const Color(0xFFC62828),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 8),

                    // Acceptance status text
                    Text(
                      _result!.accepted
                          ? '✅ Meets research confidence gate (≥ 52%)'
                          : '⚠️ Uncertain — retake photo or request expert grading',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        color: _result!.accepted
                            ? const Color(0xFF2E7D32)
                            : const Color(0xFFC62828),
                      ),
                    ),

                    const Divider(height: 24),

                    // Timings Grid (3 Cards)
                    Row(
                      children: [
                        _buildTimingCard('Preprocessing', '${_result!.preprocessMs} ms', Icons.image_outlined),
                        const SizedBox(width: 8),
                        _buildTimingCard('ONNX Infer', '${_result!.inferenceMs} ms', Icons.bolt),
                        const SizedBox(width: 8),
                        _buildTimingCard('Total Latency', '${_result!.totalMs} ms', Icons.timer_outlined, highlight: true),
                      ],
                    ),

                    const Divider(height: 24),

                    // 4-Class Probabilities
                    const Text(
                      'Class Probability Distribution',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),

                    ...List.generate(4, (i) {
                      final c = kClasses[i];
                      final prob = _result!.probs[i];
                      final isTop = c == _result!.label;
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 6),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  kClassDisplayNames[c] ?? c,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                                    color: isTop ? const Color(0xFF1B4D3E) : Colors.black87,
                                  ),
                                ),
                                Text(
                                  '${(prob * 100).toStringAsFixed(1)}%',
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: isTop ? FontWeight.bold : FontWeight.w500,
                                    color: isTop ? const Color(0xFF1B4D3E) : Colors.black87,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 3),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value: prob,
                                minHeight: 6,
                                backgroundColor: Colors.grey.shade200,
                                color: isTop ? const Color(0xFF1B4D3E) : const Color(0xFFB3541E),
                              ),
                            ),
                          ],
                        ),
                      );
                    }),

                    const Divider(height: 24),

                    // Expert Ground-Truth Verification
                    const Text(
                      'Expert Ground-Truth Grade (for Accuracy Validation):',
                      style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 6),

                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: kClasses.map((g) {
                        final isSelected = _expertGrade == g;
                        return ChoiceChip(
                          label: Text(
                            kClassDisplayNames[g]?.split(' ').first ?? g,
                            style: TextStyle(
                              fontSize: 11.5,
                              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                              color: isSelected ? Colors.white : Colors.black87,
                            ),
                          ),
                          selected: isSelected,
                          selectedColor: const Color(0xFF1B4D3E),
                          onSelected: (selected) {
                            setState(() {
                              _expertGrade = selected ? g : '';
                            });
                          },
                        );
                      }).toList(),
                    ),

                    if (_expertGrade.isNotEmpty) ...[
                      const SizedBox(height: 6),
                      Text(
                        _expertGrade == _result!.label
                            ? '✅ Expert grade matches model prediction!'
                            : 'ℹ️ Discrepancy logged for research accuracy tracking.',
                        style: TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w500,
                          color: _expertGrade == _result!.label
                              ? const Color(0xFF2E7D32)
                              : const Color(0xFFE65100),
                        ),
                      ),
                    ],

                    const SizedBox(height: 16),

                    // Save Button
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _savePrediction,
                        icon: const Icon(Icons.save_outlined),
                        label: const Text('Save Prediction to CSV'),
                        style: FilledButton.styleFrom(
                          backgroundColor: const Color(0xFF1B4D3E),
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
          ],

          // Research Accuracy & CSV Export Card
          Card(
            color: Colors.white,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Research Evaluation',
                        style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          '$_savedRowCount records saved',
                          style: TextStyle(fontSize: 11, color: Colors.grey.shade700),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Expert-verified Accuracy:',
                              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              _evaluatedCount == 0
                                  ? 'No verified samples yet'
                                  : '${((_correctCount / _evaluatedCount) * 100).toStringAsFixed(1)}% ($_correctCount / $_evaluatedCount)',
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1B4D3E),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  if (_csvFilePath != null)
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _shareCsv,
                        icon: const Icon(Icons.share, color: Color(0xFF1B4D3E)),
                        label: const Text(
                          'Share CSV Report',
                          style: TextStyle(color: Color(0xFF1B4D3E), fontWeight: FontWeight.bold),
                        ),
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Color(0xFF1B4D3E)),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),
                    )
                  else
                    Text(
                      'Save at least one prediction to generate and share the CSV report.',
                      style: TextStyle(fontSize: 11.5, color: Colors.grey.shade500),
                    ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 14),

          // Footnote / Research details
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Text(
              'Input: 384x384 NCHW | Temp: 1.457 | Gate: 0.52 | Yolov8s-cls\n'
              'Predictions are research estimates. Offline inference runs entirely on-device.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildTimingCard(String title, String value, IconData icon, {bool highlight = false}) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
        decoration: BoxDecoration(
          color: highlight ? const Color(0xFFE8F5E9) : const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: highlight ? const Color(0xFFA5D6A7) : Colors.grey.shade300,
          ),
        ),
        child: Column(
          children: [
            Icon(
              icon,
              size: 16,
              color: highlight ? const Color(0xFF2E7D32) : Colors.grey.shade700,
            ),
            const SizedBox(height: 2),
            Text(
              value,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.bold,
                color: highlight ? const Color(0xFF2E7D32) : Colors.black87,
              ),
            ),
            Text(
              title,
              style: TextStyle(
                fontSize: 9.5,
                color: highlight ? const Color(0xFF2E7D32) : Colors.grey.shade600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
