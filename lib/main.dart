import 'dart:convert' show utf8, jsonDecode;
import 'dart:io' show File, Directory;
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
import 'research_prediction.dart';
export 'research_prediction.dart';
import 'widgets/scan_animation.dart';
import 'widgets/bulk_analysis_view.dart';

const Map<String, String> kClassDisplayNames = {
  'extra_special': 'Extra Special (Alba / C5)',
  'high_quality': 'High Quality (C5 / M5)',
  'medium_quality': 'Medium Quality (M4 / H1)',
  'low_quality': 'Low Quality (H2 / Quillings)',
};

const String kInputName = 'images';
const String kModelAssetPath = 'assets/models/frozen_yolov8s_fp32_384.onnx';

class EdgeEngine {
  bool biasEnabled = false;
  double selectedBias = extraSpecialBias;
  String modelVersion = 'unidentified';
  String modelSha256 = '';
  bool outputsProbabilities = true;
  bool verifiedCleanModel = false;

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
    final bytes = await rootBundle.load(kModelAssetPath);
    modelSha256 = sha256.convert(bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes)).toString();
    // Bind calibration to the exact model, so replacing an asset cannot silently
    // carry a validation-selected correction over to a different experiment.
    final manifest = jsonDecode(await rootBundle.loadString('assets/models/deployment_manifest.json')) as Map<String, dynamic>;
    if (manifest['model_sha256'] != modelSha256 ||
        manifest['input_name'] != kInputName ||
        manifest['class_order'].toString() != kClasses.toString()) {
      throw StateError('Deployment manifest does not match the bundled model. Verify its identity and class order.');
    }
    final bias = manifest['extra_special_bias'];
    if (bias is! num || !bias.isFinite || bias.abs() > 20) {
      throw StateError('Invalid deployment ES bias.');
    }
    selectedBias = bias.toDouble();
    modelVersion = manifest['model_version'] as String;
    outputsProbabilities = manifest['output_type'] == 'probabilities';
    if (!outputsProbabilities && manifest['output_type'] != 'logits') {
      throw StateError('Deployment output type must be logits or probabilities.');
    }
    verifiedCleanModel = manifest['dataset'] == 'clean_ceylon_cinnamon';
    biasEnabled = verifiedCleanModel && manifest['bias_default_enabled'] == true;
    final options = OrtSessionOptions(
      providers: [kIsWeb ? OrtProvider.WEB_ASSEMBLY : OrtProvider.CPU],
    );
    // Flutter Web serves declared assets under an additional assets/ prefix;
    // the plugin forwards web URLs directly rather than resolving asset keys.
    final session = kIsWeb
        ? await _runtime.createSession('assets/$kModelAssetPath', options: options)
        : await _runtime.createSessionFromAsset(kModelAssetPath, options: options);
    if (session.inputNames.toString() != '[images]' || session.outputNames.toString() != '[output0]') {
      await session.close();
      throw StateError('Unexpected model input/output names.');
    }
    _session = session;
  }

  Float32List _tensor(img.Image source) {
    // Match training geometry and channel scale: stretching bark or changing
    // padding/normalization would shift the classifier's input distribution.
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

    final useBias = biasEnabled;
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

    // Keep the full SHA-256 image identity for deduplicated research evaluation
    final fingerprint = sha256.convert(bytes).toString();

    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Image could not be decoded. Ensure it is a valid JPEG/PNG.');
    }

    final nchw = _tensor(decoded);
    final prepMs = prepTimer.elapsedMilliseconds;

    final ortInput = await OrtValue.fromList(nchw.toList(), [1, 3, 384, 384]);
    final inferTimer = Stopwatch()..start();

    Map<String, OrtValue>? output;
    try {
      output = await _session!.run({kInputName: ortInput});
      if (output.values.isEmpty) {
        throw StateError('ONNX model returned empty output tensor.');
      }
      final tensor = output['output0'];
      if (tensor == null || tensor.shape.toString() != '[1, 4]') {
        throw StateError('Expected model output0 shape [1, 4].');
      }
      final raw = await tensor.asList();
      inferTimer.stop();
      final postTimer = Stopwatch()..start();
      final values = _flatten(raw);
      if (values.length != 4) {
        throw StateError(
          'Unexpected ONNX output shape: received ${values.length} values (expected 4).',
        );
      }

      // The checked deployment graph ends in Softmax. Preserve its original
      // probabilities; the old experiment's temperature must not be reused.
      final original = modelProbabilities(values, outputsProbabilities: outputsProbabilities);
      final corrected = applyExtraSpecialBias(original, bias: selectedBias);
      postTimer.stop();
      totalTimer.stop();
      return Result(
        rawProbs: original,
        correctedProbs: corrected,
        biasEnabled: useBias,
        bias: selectedBias,
        modelVersion: modelVersion,
        modelSha256: modelSha256,
        runtimePlatform: kIsWeb ? 'web' : defaultTargetPlatform.name,
        inferenceProvider: kIsWeb ? 'WASM' : 'CPU',
        preprocessMs: prepMs,
        inferenceMs: inferTimer.elapsedMilliseconds,
        postprocessingMs: postTimer.elapsedMilliseconds,
        totalMs: totalTimer.elapsedMilliseconds,
        imageFingerprint: fingerprint,
      );
    } finally {
      for (final tensor in output?.values ?? <OrtValue>[]) {
        await tensor.dispose();
      }
      await ortInput.dispose();
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
      title: 'CinnaGrade AI',
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
  final ResearchEvaluation _evaluation = ResearchEvaluation();
  final Set<String> _savedPredictions = {};
  int _savedRowCount = 0;
  String? _csvFilePath;

  final List<String> _csvRows = [Result.csvHeader];

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
            _scanStatusText = 'Validating probabilities and applying selected ES correction...';
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
    // Repeated taps must not masquerade as independent research samples.
    final saveKey = '${r.timestamp.toIso8601String()}:${r.imageFingerprint}:${r.biasEnabled}:$_expertGrade';
    if (!_savedPredictions.add(saveKey)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('This prediction and expert label are already saved.')));
      return;
    }
    _evaluation.record(r, _expertGrade);
    final filename = _imageName ?? 'image.png';
    _csvRows.add(r.csvRow(filename, expertGrade: _expertGrade));

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

  Future<void> _downloadOrSaveCsv() async {
    if (_csvRows.length <= 1) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No predictions saved to download yet.'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    final csvContent = '${_csvRows.join('\n')}\n';
    final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    final fileName = 'cinnamon_edge_predictions_$timestamp.csv';

    try {
      if (kIsWeb) {
        final bytes = utf8.encode(csvContent);
        final xFile = XFile.fromData(
          Uint8List.fromList(bytes),
          mimeType: 'text/csv',
          name: fileName,
        );
        await Share.shareXFiles(
          [xFile],
          text: 'CinnaGrade AI Research Predictions (Field Trial)',
          fileNameOverrides: [fileName],
        );

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.check_circle_outline, color: Colors.white),
                  const SizedBox(width: 8),
                  Expanded(child: Text('Downloaded "$fileName"')),
                ],
              ),
              backgroundColor: const Color(0xFF1B4D3E),
              duration: const Duration(seconds: 3),
            ),
          );
        }
        return;
      }

      // On Mobile / Desktop: Save to Downloads folder or App Documents
      Directory? targetDir;
      try {
        targetDir = await getDownloadsDirectory();
      } catch (_) {}
      targetDir ??= await getApplicationDocumentsDirectory();

      final file = File('${targetDir.path}/$fileName');
      await file.writeAsString(csvContent);
      _csvFilePath = file.path;

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle_outline, color: Colors.white),
                const SizedBox(width: 8),
                Expanded(child: Text('Saved to: ${file.path}')),
              ],
            ),
            backgroundColor: const Color(0xFF1B4D3E),
            duration: const Duration(seconds: 4),
            action: SnackBarAction(
              label: 'SHARE',
              textColor: Colors.white,
              onPressed: _shareCsv,
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error saving CSV: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _shareCsv() async {
    if (_csvRows.length <= 1) return;
    final csvContent = '${_csvRows.join('\n')}\n';
    final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    final fileName = 'cinnamon_edge_predictions_$timestamp.csv';

    try {
      if (kIsWeb) {
        final bytes = utf8.encode(csvContent);
        final xFile = XFile.fromData(
          Uint8List.fromList(bytes),
          mimeType: 'text/csv',
          name: fileName,
        );
        await Share.shareXFiles(
          [xFile],
          text: 'CinnaGrade AI Research Predictions (Field Trial)',
          fileNameOverrides: [fileName],
        );
        return;
      }

      if (_csvFilePath == null || !File(_csvFilePath!).existsSync()) {
        final dir = await getApplicationDocumentsDirectory();
        final file = File('${dir.path}/$fileName');
        await file.writeAsString(csvContent);
        _csvFilePath = file.path;
      }

      await Share.shareXFiles(
        [XFile(_csvFilePath!)],
        text: 'CinnaGrade AI Research Predictions (Field Trial)',
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Share error: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  void _showCsvPreviewDialog() {
    final csvContent = '${_csvRows.join('\n')}\n';
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.72,
          maxChildSize: 0.94,
          minChildSize: 0.4,
          builder: (context, scrollController) {
            return Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'CSV Research Report Data',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF1B4D3E),
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                  Text(
                    '$_savedRowCount records logged. You can copy raw CSV text or save as a file.',
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF8F9FA),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.grey.shade300),
                      ),
                      child: SingleChildScrollView(
                        controller: scrollController,
                        scrollDirection: Axis.vertical,
                        child: SingleChildScrollView(
                          scrollDirection: Axis.horizontal,
                          child: SelectableText(
                            csvContent,
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 11,
                              color: Color(0xFF222222),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () {
                            Clipboard.setData(ClipboardData(text: csvContent));
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('CSV copied to clipboard! Paste directly into Excel or Sheets.'),
                                backgroundColor: Color(0xFF1B4D3E),
                                duration: Duration(seconds: 2),
                              ),
                            );
                            Navigator.pop(context);
                          },
                          icon: const Icon(Icons.copy, size: 18),
                          label: const Text('Copy Data'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: const Color(0xFF1B4D3E),
                            side: const BorderSide(color: Color(0xFF1B4D3E)),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: () {
                            Navigator.pop(context);
                            _downloadOrSaveCsv();
                          },
                          icon: const Icon(Icons.download, size: 18),
                          label: const Text('Save File'),
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
                ],
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
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
                  'CinnaGrade AI',
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
                                kIsWeb ? 'WEB ONNX READY' : 'OFFLINE READY (REAL ONNX)',
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
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(48),
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFFF0F3EF),
              borderRadius: BorderRadius.circular(12),
            ),
            child: TabBar(
              indicatorSize: TabBarIndicatorSize.tab,
              indicator: BoxDecoration(
                color: const Color(0xFF1B4D3E),
                borderRadius: BorderRadius.circular(10),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF1B4D3E).withValues(alpha: 0.2),
                    blurRadius: 4,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              labelColor: Colors.white,
              unselectedLabelColor: const Color(0xFF4A554D),
              labelStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
              unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
              dividerColor: Colors.transparent,
              tabs: const [
                Tab(
                  iconMargin: EdgeInsets.zero,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.camera_alt, size: 16),
                      SizedBox(width: 8),
                      Text('Single Tree Scan'),
                    ],
                  ),
                ),
                Tab(
                  iconMargin: EdgeInsets.zero,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.auto_awesome_motion, size: 16),
                      SizedBox(width: 8),
                      Text('Batch Analysis (Bulk)'),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      body: TabBarView(
        children: [
          ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        children: [
          SwitchListTile.adaptive(
            title: const Text('Extra Special Correction'),
            subtitle: Text('ES log-bias = ${_engine.selectedBias}. ${_engine.biasEnabled ? "ON" : "OFF"}. Batch records retain the setting used at inference.'),
            value: _engine.biasEnabled,
            onChanged: !_isModelLoaded || _isProcessing ? null : (enabled) {
              setState(() {
                _engine.biasEnabled = enabled;
                _result = _result?.withBiasEnabled(enabled);
              });
            },
          ),
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
                            'REAL_ONNX_RUNTIME (Fingerprint: ${_result!.imageFingerprint.substring(0, 16)})',
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

                    Text('Original: ${kClassDisplayNames[_result!.rawLabel]} — ${(_result!.rawConfidence * 100).toStringAsFixed(1)}%'),
                    Text('Corrected: ${kClassDisplayNames[_result!.correctedLabel]} — ${(_result!.correctedConfidence * 100).toStringAsFixed(1)}%'),
                    Text('Final grade uses ${_result!.biasEnabled ? "corrected" : "original"} probabilities. Confidence is not accuracy.'),
                    const SizedBox(height: 8),
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
                      'Class probabilities: Original / Corrected',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),

                    ...List.generate(4, (i) {
                      final c = kClasses[i];
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
                                  '${(_result!.rawProbs[i] * 100).toStringAsFixed(1)}% / ${(_result!.correctedProbs[i] * 100).toStringAsFixed(1)}%',
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
                                value: _result!.probs[i],
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
                              'Paired evaluation on unique expert-labelled images:',
                              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                            ),
                            const SizedBox(height: 2),
                            Text(
                              _evaluation.report(),
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
                  if (_savedRowCount > 0 || _csvRows.length > 1) ...[
                    Row(
                      children: [
                        Expanded(
                          child: FilledButton.icon(
                            onPressed: _downloadOrSaveCsv,
                            icon: const Icon(Icons.file_download, size: 18),
                            label: const Text(
                              'Export Research CSV',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            style: FilledButton.styleFrom(
                              backgroundColor: const Color(0xFF1B4D3E),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _shareCsv,
                            icon: const Icon(Icons.share, size: 18, color: Color(0xFF1B4D3E)),
                            label: const Text(
                              'Share CSV',
                              style: TextStyle(color: Color(0xFF1B4D3E), fontWeight: FontWeight.bold),
                            ),
                            style: OutlinedButton.styleFrom(
                              side: const BorderSide(color: Color(0xFF1B4D3E)),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    SizedBox(
                      width: double.infinity,
                      child: TextButton.icon(
                        onPressed: _showCsvPreviewDialog,
                        icon: const Icon(Icons.copy_all, size: 16),
                        label: const Text('View & Copy Raw CSV Data'),
                        style: TextButton.styleFrom(
                          foregroundColor: const Color(0xFF1B4D3E),
                        ),
                      ),
                    ),
                  ] else ...[
                    Text(
                      'Save at least one prediction to generate, download, and share the CSV report.',
                      style: TextStyle(fontSize: 11.5, color: Colors.grey.shade500),
                    ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 14),

          // Footnote / Research details
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Text(
              'Input: 384x384 NCHW | ES log-bias: 0.6 | Gate: 0.52 | YOLOv8s-cls\n'
              'Predictions are research estimates. Offline inference runs entirely on-device.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),

      // Tab 2: Batch Analysis (Bulk Inspection for 10 - 900+ Trees)
      BulkAnalysisView(
        engine: _engine,
        isModelLoaded: _isModelLoaded,
        onAppendToGlobalCsv: (row) {
          _csvRows.add(row);
          _savedRowCount++;
          if (mounted) setState(() {});
        },
      ),
    ],
  ),
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
