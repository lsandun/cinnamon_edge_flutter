import 'dart:convert' show utf8;
import 'dart:io' show File, Directory;
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import '../main.dart';

/// Represents a single image result within a batch analysis job.
class BatchItem {
  final int index;
  final String fileName;
  final String filePath;
  final Uint8List? thumbnailBytes;
  final Result? result;
  final String? errorMessage;
  final DateTime processedAt;

  BatchItem({
    required this.index,
    required this.fileName,
    required this.filePath,
    this.thumbnailBytes,
    this.result,
    this.errorMessage,
    required this.processedAt,
  });

  bool get isSuccess => result != null;
  bool get isAccepted => result != null && result!.accepted;
}

enum BatchFilter { all, extraSpecial, highQuality, mediumQuality, lowQuality, flagged }
enum BatchViewMode { grid, table }

class BulkAnalysisView extends StatefulWidget {
  final EdgeEngine engine;
  final bool isModelLoaded;
  final Function(String csvRow)? onAppendToGlobalCsv;

  const BulkAnalysisView({
    super.key,
    required this.engine,
    required this.isModelLoaded,
    this.onAppendToGlobalCsv,
  });

  @override
  State<BulkAnalysisView> createState() => _BulkAnalysisViewState();
}

class _BulkAnalysisViewState extends State<BulkAnalysisView> {
  final ImagePicker _picker = ImagePicker();
  final List<BatchItem> _batchItems = [];
  final TextEditingController _searchController = TextEditingController();

  bool _isProcessing = false;
  bool _isPaused = false;
  bool _isCancelled = false;

  int _currentIndex = 0;
  int _totalQueued = 0;
  String _currentFileName = '';

  Stopwatch _batchStopwatch = Stopwatch();
  BatchFilter _activeFilter = BatchFilter.all;
  BatchViewMode _viewMode = BatchViewMode.grid;
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    _searchController.addListener(() {
      setState(() {
        _searchQuery = _searchController.text.trim().toLowerCase();
      });
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  // --- Statistics Calculation ---
  int get _successCount => _batchItems.where((i) => i.isSuccess).length;
  int get _acceptedCount => _batchItems.where((i) => i.isAccepted).length;
  int get _flaggedCount => _batchItems.where((i) => i.isSuccess && !i.isAccepted).length;

  int get _extraCount => _batchItems.where((i) => i.result?.label == 'extra_special').length;
  int get _highCount => _batchItems.where((i) => i.result?.label == 'high_quality').length;
  int get _mediumCount => _batchItems.where((i) => i.result?.label == 'medium_quality').length;
  int get _lowCount => _batchItems.where((i) => i.result?.label == 'low_quality').length;

  double get _avgLatencyMs {
    final list = _batchItems.where((i) => i.isSuccess).map((i) => i.result!.totalMs).toList();
    if (list.isEmpty) return 0.0;
    return list.reduce((a, b) => a + b) / list.length;
  }

  double get _throughputFps {
    final elapsedSec = _batchStopwatch.elapsedMilliseconds / 1000.0;
    if (elapsedSec <= 0 || _currentIndex <= 0) return 0.0;
    return _currentIndex / elapsedSec;
  }

  String get _etaString {
    if (!_isProcessing || _currentIndex <= 0 || _totalQueued <= 0) return '';
    final remaining = _totalQueued - _currentIndex;
    final fps = _throughputFps;
    if (fps <= 0.1) return 'Calculating...';
    final sec = (remaining / fps).round();
    if (sec < 60) return '~${sec}s remaining';
    final min = sec ~/ 60;
    final remSec = sec % 60;
    return '~${min}m ${remSec}s remaining';
  }

  List<BatchItem> get _filteredItems {
    return _batchItems.where((item) {
      // Filter by Grade / Status
      if (_activeFilter == BatchFilter.extraSpecial && item.result?.label != 'extra_special') return false;
      if (_activeFilter == BatchFilter.highQuality && item.result?.label != 'high_quality') return false;
      if (_activeFilter == BatchFilter.mediumQuality && item.result?.label != 'medium_quality') return false;
      if (_activeFilter == BatchFilter.lowQuality && item.result?.label != 'low_quality') return false;
      if (_activeFilter == BatchFilter.flagged && (item.isAccepted || !item.isSuccess)) return false;

      // Filter by Search Query
      if (_searchQuery.isNotEmpty) {
        final matchesName = item.fileName.toLowerCase().contains(_searchQuery);
        final matchesFp = item.result?.imageFingerprint.toLowerCase().contains(_searchQuery) ?? false;
        final matchesGrade = item.result?.label.toLowerCase().contains(_searchQuery) ?? false;
        if (!matchesName && !matchesFp && !matchesGrade) return false;
      }
      return true;
    }).toList();
  }

  // --- Multi-Image Picker & Batch Processor ---
  Future<void> _pickAndProcessImages() async {
    if (!widget.isModelLoaded) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('ONNX Model is still loading. Please wait until engine is ready.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final List<XFile> pickedFiles = await _picker.pickMultiImage(
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 92,
      );

      if (pickedFiles.isEmpty) return;

      setState(() {
        _isProcessing = true;
        _isPaused = false;
        _isCancelled = false;
        _totalQueued = pickedFiles.length;
        _currentIndex = 0;
        _currentFileName = '';
      });

      _batchStopwatch = Stopwatch()..start();

      for (int i = 0; i < pickedFiles.length; i++) {
        if (_isCancelled) break;

        // Check for pause
        while (_isPaused && !_isCancelled) {
          await Future.delayed(const Duration(milliseconds: 200));
        }
        if (_isCancelled) break;

        final file = pickedFiles[i];
        final fileName = file.name.isNotEmpty
            ? file.name
            : file.path.split(RegExp(r'[/\\]')).last;

        setState(() {
          _currentIndex = i + 1;
          _currentFileName = fileName;
        });

        try {
          final bytes = await file.readAsBytes();

          // Generate lightweight 80x80 thumbnail to conserve RAM for 900+ images
          Uint8List? thumbBytes;
          try {
            final decoded = img.decodeImage(bytes);
            if (decoded != null) {
              final thumb = img.copyResize(decoded, width: 80, height: 80, interpolation: img.Interpolation.linear);
              thumbBytes = Uint8List.fromList(img.encodeJpg(thumb, quality: 65));
            }
          } catch (_) {}

          // Run Real ONNX Inference
          final result = await widget.engine.predict(
            file.path,
            imageBytes: bytes,
          );

          final item = BatchItem(
            index: _batchItems.length + 1,
            fileName: fileName,
            filePath: file.path,
            thumbnailBytes: thumbBytes,
            result: result,
            processedAt: DateTime.now(),
          );

          if (mounted) {
            setState(() {
              _batchItems.add(item);
            });
          }

          // Optionally notify global logger
          if (widget.onAppendToGlobalCsv != null) {
            final csvLine = [
              DateTime.now().toUtc().toIso8601String(),
              result.imageFingerprint,
              fileName,
              'REAL_ONNX_RUNTIME',
              result.label,
              result.confidence.toStringAsFixed(6),
              result.accepted,
              result.preprocessMs,
              result.inferenceMs,
              result.totalMs,
              '',
              '',
              ...result.probs.map((p) => p.toStringAsFixed(6)),
            ].map((v) => '"${v.toString().replaceAll('"', '""')}"').join(',');
            widget.onAppendToGlobalCsv!(csvLine);
          }
        } catch (e) {
          final errorItem = BatchItem(
            index: _batchItems.length + 1,
            fileName: fileName,
            filePath: file.path,
            errorMessage: e.toString(),
            processedAt: DateTime.now(),
          );
          if (mounted) {
            setState(() {
              _batchItems.add(errorItem);
            });
          }
        }

        // Yield to event loop to keep UI smooth at 60fps
        await Future.delayed(Duration.zero);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Batch selection error: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      _batchStopwatch.stop();
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _isPaused = false;
        });
      }
    }
  }

  void _pauseResumeBatch() {
    setState(() {
      _isPaused = !_isPaused;
    });
  }

  void _cancelBatch() {
    setState(() {
      _isCancelled = true;
      _isProcessing = false;
      _isPaused = false;
    });
  }

  void _clearBatch() {
    if (_isProcessing) return;
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear Batch Results?'),
        content: Text('Are you sure you want to clear all ${_batchItems.length} scanned tree records?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              setState(() {
                _batchItems.clear();
                _currentIndex = 0;
                _totalQueued = 0;
              });
            },
            style: FilledButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('Clear All'),
          ),
        ],
      ),
    );
  }

  // --- CSV Export for Batch Analysis ---
  String _generateBatchCsvContent() {
    final rows = [
      'timestamp_utc,image_fingerprint,image,runtime_mode,grade,confidence,accepted,preprocess_ms,inference_ms,total_ms,prob_extra,prob_high,prob_low,prob_medium'
    ];

    String esc(dynamic s) => '"${s.toString().replaceAll('"', '""')}"';

    for (final item in _batchItems) {
      if (item.result != null) {
        final r = item.result!;
        rows.add([
          esc(item.processedAt.toUtc().toIso8601String()),
          esc(r.imageFingerprint),
          esc(item.fileName),
          'REAL_ONNX_RUNTIME',
          esc(r.label),
          r.confidence.toStringAsFixed(6),
          r.accepted,
          r.preprocessMs.toString(),
          r.inferenceMs.toString(),
          r.totalMs.toString(),
          ...r.probs.map((v) => v.toStringAsFixed(6)),
        ].join(','));
      }
    }
    return '${rows.join('\n')}\n';
  }

  Future<void> _exportBatchCsv() async {
    if (_batchItems.isEmpty) return;

    final csvContent = _generateBatchCsvContent();
    final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    final fileName = 'cinnamon_batch_analysis_${_batchItems.length}_samples_$timestamp.csv';

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
          text: 'Cinnamon Edge Batch Research Report (${_batchItems.length} Trees)',
          fileNameOverrides: [fileName],
        );
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Downloaded "$fileName" to browser Downloads!'),
              backgroundColor: const Color(0xFF1B4D3E),
            ),
          );
        }
        return;
      }

      Directory? targetDir;
      try {
        targetDir = await getDownloadsDirectory();
      } catch (_) {}
      targetDir ??= await getApplicationDocumentsDirectory();

      final file = File('${targetDir.path}/$fileName');
      await file.writeAsString(csvContent);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Saved: ${file.path}'),
            backgroundColor: const Color(0xFF1B4D3E),
            action: SnackBarAction(
              label: 'SHARE',
              textColor: Colors.white,
              onPressed: () {
                Share.shareXFiles([XFile(file.path)], text: 'Cinnamon Batch Report');
              },
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Export error: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  void _showItemDetailModal(BatchItem item) {
    if (item.result == null) return;
    final r = item.result!;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Container(
                  width: 44,
                  height: 4,
                  decoration: BoxDecoration(color: Colors.grey.shade300, borderRadius: BorderRadius.circular(2)),
                ),
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  if (item.thumbnailBytes != null)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Image.memory(item.thumbnailBytes!, width: 64, height: 64, fit: BoxFit.cover),
                    )
                  else
                    Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(color: Colors.grey.shade200, borderRadius: BorderRadius.circular(10)),
                      child: const Icon(Icons.image, color: Colors.grey),
                    ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          item.fileName,
                          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(4)),
                          child: Text(
                            'FP: ${r.imageFingerprint}',
                            style: const TextStyle(fontSize: 10.5, fontFamily: 'monospace', color: Colors.black87),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Latency: ${r.totalMs} ms (${r.preprocessMs} prep + ${r.inferenceMs} infer)',
                          style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              const Divider(height: 1),
              const SizedBox(height: 14),

              // Predicted Grade & Acceptance Status
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('PREDICTED GRADE', style: TextStyle(fontSize: 10.5, letterSpacing: 0.8, color: Colors.grey.shade600, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 2),
                      Text(kClassDisplayNames[r.label] ?? r.label, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF1B4D3E))),
                    ],
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: r.accepted ? const Color(0xFFE8F5E9) : const Color(0xFFFFEBEE),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: r.accepted ? const Color(0xFF81C784) : const Color(0xFFE57373)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text('${(r.confidence * 100).toStringAsFixed(1)}%', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: r.accepted ? const Color(0xFF2E7D32) : const Color(0xFFC62828))),
                        Text(r.accepted ? 'Accepted' : 'Flagged (<52%)', style: TextStyle(fontSize: 9.5, color: r.accepted ? const Color(0xFF2E7D32) : const Color(0xFFC62828))),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),

              // Probabilities
              const Text('Class Probability Breakdown:', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
              const SizedBox(height: 8),
              ...List.generate(4, (idx) {
                final c = kClasses[idx];
                final prob = r.probs[idx];
                final isTop = c == r.label;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(kClassDisplayNames[c] ?? c, style: TextStyle(fontSize: 11.5, fontWeight: isTop ? FontWeight.bold : FontWeight.normal)),
                          Text('${(prob * 100).toStringAsFixed(1)}%', style: TextStyle(fontSize: 11.5, fontWeight: isTop ? FontWeight.bold : FontWeight.normal)),
                        ],
                      ),
                      const SizedBox(height: 2),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: LinearProgressIndicator(
                          value: prob,
                          minHeight: 5,
                          backgroundColor: Colors.grey.shade200,
                          color: isTop ? const Color(0xFF1B4D3E) : const Color(0xFFB3541E),
                        ),
                      ),
                    ],
                  ),
                );
              }),
              const SizedBox(height: 16),
              SizedBox(
                width: double.infinity,
                child: FilledButton(
                  onPressed: () => Navigator.pop(ctx),
                  style: FilledButton.styleFrom(backgroundColor: const Color(0xFF1B4D3E)),
                  child: const Text('Close Details'),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Color _getGradeColor(String label) {
    switch (label) {
      case 'extra_special':
        return const Color(0xFFD4AF37); // Gold
      case 'high_quality':
        return const Color(0xFF2E7D32); // Emerald Green
      case 'medium_quality':
        return const Color(0xFF0288D1); // Ocean Blue
      case 'low_quality':
        return const Color(0xFFE65100); // Amber Orange
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    final filtered = _filteredItems;

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      children: [
        // --- Header Upload & Batch Action Card ---
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF1B4D3E), Color(0xFF296350)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF1B4D3E).withValues(alpha: 0.25),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(Icons.auto_awesome_motion, color: Colors.white, size: 28),
                  ),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Batch Field Inspection',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 2),
                        Text(
                          'Scan 10 to 900+ cinnamon tree bark photos at once with offline AI.',
                          style: TextStyle(color: Color(0xFFD4E9DE), fontSize: 11.5),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _pickAndProcessImages,
                      icon: const Icon(Icons.add_photo_alternate, size: 20),
                      label: Text(
                        _isProcessing ? 'Processing Batch...' : 'Select Photos (තොග ඡායාරූප)',
                        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13.5),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: const Color(0xFF1B4D3E),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                        elevation: 0,
                      ),
                    ),
                  ),
                  if (_batchItems.isNotEmpty && !_isProcessing) ...[
                    const SizedBox(width: 8),
                    IconButton.filled(
                      onPressed: _exportBatchCsv,
                      tooltip: 'Export Batch CSV',
                      icon: const Icon(Icons.file_download, color: Colors.white),
                      style: IconButton.styleFrom(backgroundColor: Colors.white.withValues(alpha: 0.25)),
                    ),
                    const SizedBox(width: 6),
                    IconButton.filled(
                      onPressed: _clearBatch,
                      tooltip: 'Clear Batch',
                      icon: const Icon(Icons.delete_sweep, color: Colors.white),
                      style: IconButton.styleFrom(backgroundColor: Colors.white.withValues(alpha: 0.25)),
                    ),
                  ],
                ],
              ),
            ],
          ),
        ),

        const SizedBox(height: 12),

        // --- Live Batch Telemetry & Progress Card (When Processing) ---
        if (_isProcessing) ...[
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFF00E676).withValues(alpha: 0.4)),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF00E676).withValues(alpha: 0.08),
                  blurRadius: 10,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF1B4D3E)),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isPaused ? 'BATCH PAUSED' : 'AI BATCH INFERENCE ACTIVE',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.6,
                            color: _isPaused ? Colors.orange.shade800 : const Color(0xFF1B4D3E),
                          ),
                        ),
                      ],
                    ),
                    Text(
                      '$_currentIndex / $_totalQueued (${((_currentIndex / math.max(1, _totalQueued)) * 100).toInt()}%)',
                      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13, color: Color(0xFF1B4D3E)),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _totalQueued > 0 ? (_currentIndex / _totalQueued) : 0,
                    minHeight: 8,
                    backgroundColor: Colors.grey.shade200,
                    valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF00E676)),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        'Current: $_currentFileName',
                        style: TextStyle(fontSize: 11, color: Colors.grey.shade700),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Text(
                      '${_throughputFps.toStringAsFixed(1)} img/s | $_etaString',
                      style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.grey.shade800),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton.icon(
                      onPressed: _pauseResumeBatch,
                      icon: Icon(_isPaused ? Icons.play_arrow : Icons.pause, size: 16),
                      label: Text(_isPaused ? 'Resume' : 'Pause'),
                      style: TextButton.styleFrom(foregroundColor: const Color(0xFF1B4D3E)),
                    ),
                    const SizedBox(width: 8),
                    TextButton.icon(
                      onPressed: _cancelBatch,
                      icon: const Icon(Icons.stop, size: 16, color: Colors.red),
                      label: const Text('Stop Batch', style: TextStyle(color: Colors.red)),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
        ],

        // --- Live Batch Analytics Dashboard ---
        if (_batchItems.isNotEmpty) ...[
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Batch Telemetry & Quality Distribution',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13.5),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: const Color(0xFFE8F5E9),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '${((_acceptedCount / math.max(1, _successCount)) * 100).toStringAsFixed(1)}% Accepted',
                        style: const TextStyle(fontSize: 10.5, fontWeight: FontWeight.bold, color: Color(0xFF2E7D32)),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                // 4 Grade Counter Tiles
                Row(
                  children: [
                    _buildGradeTile('Extra', _extraCount, const Color(0xFFD4AF37)),
                    const SizedBox(width: 6),
                    _buildGradeTile('High', _highCount, const Color(0xFF2E7D32)),
                    const SizedBox(width: 6),
                    _buildGradeTile('Medium', _mediumCount, const Color(0xFF0288D1)),
                    const SizedBox(width: 6),
                    _buildGradeTile('Low', _lowCount, const Color(0xFFE65100)),
                  ],
                ),

                const SizedBox(height: 12),

                // Proportional Color Stack Bar
                if (_successCount > 0)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: SizedBox(
                      height: 10,
                      child: Row(
                        children: [
                          if (_extraCount > 0) Expanded(flex: _extraCount, child: Container(color: const Color(0xFFD4AF37))),
                          if (_highCount > 0) Expanded(flex: _highCount, child: Container(color: const Color(0xFF2E7D32))),
                          if (_mediumCount > 0) Expanded(flex: _mediumCount, child: Container(color: const Color(0xFF0288D1))),
                          if (_lowCount > 0) Expanded(flex: _lowCount, child: Container(color: const Color(0xFFE65100))),
                        ],
                      ),
                    ),
                  ),

                const SizedBox(height: 10),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Avg Latency: ${_avgLatencyMs.toStringAsFixed(0)} ms / tree', style: TextStyle(fontSize: 11.5, color: Colors.grey.shade600)),
                    Text('Total Scanned: ${_batchItems.length} images', style: TextStyle(fontSize: 11.5, fontWeight: FontWeight.bold, color: Colors.grey.shade800)),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),

          // --- Search & Filter Bar ---
          Row(
            children: [
              Expanded(
                child: SizedBox(
                  height: 38,
                  child: TextField(
                    controller: _searchController,
                    decoration: InputDecoration(
                      hintText: 'Search image name or FP...',
                      hintStyle: TextStyle(fontSize: 12, color: Colors.grey.shade500),
                      prefixIcon: const Icon(Icons.search, size: 18),
                      suffixIcon: _searchQuery.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, size: 16),
                              onPressed: () => _searchController.clear(),
                            )
                          : null,
                      filled: true,
                      fillColor: Colors.white,
                      contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 10),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide(color: Colors.grey.shade300)),
                      enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide(color: Colors.grey.shade300)),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              // View Toggle
              Container(
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), border: Border.all(color: Colors.grey.shade300)),
                child: Row(
                  children: [
                    IconButton(
                      icon: Icon(Icons.grid_view, size: 18, color: _viewMode == BatchViewMode.grid ? const Color(0xFF1B4D3E) : Colors.grey),
                      onPressed: () => setState(() => _viewMode = BatchViewMode.grid),
                      tooltip: 'Grid View',
                    ),
                    IconButton(
                      icon: Icon(Icons.view_list, size: 18, color: _viewMode == BatchViewMode.table ? const Color(0xFF1B4D3E) : Colors.grey),
                      onPressed: () => setState(() => _viewMode = BatchViewMode.table),
                      tooltip: 'Table / List View',
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 8),

          // Filter Chips
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                _buildFilterChip('All (${_batchItems.length})', BatchFilter.all),
                const SizedBox(width: 6),
                _buildFilterChip('Extra ($_extraCount)', BatchFilter.extraSpecial, color: const Color(0xFFD4AF37)),
                const SizedBox(width: 6),
                _buildFilterChip('High ($_highCount)', BatchFilter.highQuality, color: const Color(0xFF2E7D32)),
                const SizedBox(width: 6),
                _buildFilterChip('Medium ($_mediumCount)', BatchFilter.mediumQuality, color: const Color(0xFF0288D1)),
                const SizedBox(width: 6),
                _buildFilterChip('Low ($_lowCount)', BatchFilter.lowQuality, color: const Color(0xFFE65100)),
                if (_flaggedCount > 0) ...[
                  const SizedBox(width: 6),
                  _buildFilterChip('Flagged ($_flaggedCount)', BatchFilter.flagged, color: Colors.red),
                ],
              ],
            ),
          ),

          const SizedBox(height: 12),

          // --- Results Explorer: Grid View or Table View ---
          if (filtered.isEmpty)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 36),
              alignment: Alignment.center,
              child: Text('No results match the current filter or search.', style: TextStyle(color: Colors.grey.shade500, fontSize: 13)),
            )
          else if (_viewMode == BatchViewMode.grid)
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 10,
                mainAxisSpacing: 10,
                childAspectRatio: 0.82,
              ),
              itemCount: filtered.length,
              itemBuilder: (ctx, idx) => _buildGridCard(filtered[idx]),
            )
          else
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: filtered.length,
              separatorBuilder: (_, __) => const SizedBox(height: 6),
              itemBuilder: (ctx, idx) => _buildTableRow(filtered[idx]),
            ),

          const SizedBox(height: 16),

          // Export Button at bottom
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _exportBatchCsv,
              icon: const Icon(Icons.file_download),
              label: Text('Download Batch CSV (${_batchItems.length} Records)'),
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFF1B4D3E),
                padding: const EdgeInsets.symmetric(vertical: 13),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
            ),
          ),
          const SizedBox(height: 20),
        ] else if (!_isProcessing) ...[
          // Empty State Prompt
          Container(
            padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Column(
              children: [
                Icon(Icons.photo_library_outlined, size: 54, color: Colors.grey.shade300),
                const SizedBox(height: 12),
                const Text(
                  'No Batch Images Loaded',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const SizedBox(height: 6),
                Text(
                  'Select multiple photos of cinnamon trees from your field camera roll. The ONNX engine will evaluate all images and prepare a comprehensive research report.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.grey.shade600, fontSize: 12.5),
                ),
                const SizedBox(height: 16),
                OutlinedButton.icon(
                  onPressed: _pickAndProcessImages,
                  icon: const Icon(Icons.folder_open, color: Color(0xFF1B4D3E)),
                  label: const Text('Pick Photos to Start', style: TextStyle(color: Color(0xFF1B4D3E), fontWeight: FontWeight.bold)),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFF1B4D3E)),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }

  // --- Sub-Widgets ---
  Widget _buildGradeTile(String title, int count, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 6),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withValues(alpha: 0.3)),
        ),
        child: Column(
          children: [
            Text('$count', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color)),
            const SizedBox(height: 2),
            Text(title, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: color), maxLines: 1),
          ],
        ),
      ),
    );
  }

  Widget _buildFilterChip(String label, BatchFilter filter, {Color? color}) {
    final isSelected = _activeFilter == filter;
    return ChoiceChip(
      label: Text(label, style: TextStyle(fontSize: 11, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal, color: isSelected ? Colors.white : Colors.black87)),
      selected: isSelected,
      selectedColor: color ?? const Color(0xFF1B4D3E),
      backgroundColor: Colors.white,
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 0),
      onSelected: (_) {
        setState(() => _activeFilter = filter);
      },
    );
  }

  Widget _buildGridCard(BatchItem item) {
    if (item.result == null) {
      return Card(
        color: const Color(0xFFFFEBEE),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, color: Colors.red),
              const SizedBox(height: 6),
              Text(item.fileName, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold), maxLines: 1),
              Text('Error processing', style: TextStyle(fontSize: 10, color: Colors.red.shade700)),
            ],
          ),
        ),
      );
    }

    final r = item.result!;
    final gradeColor = _getGradeColor(r.label);

    return InkWell(
      onTap: () => _showItemDetailModal(item),
      borderRadius: BorderRadius.circular(12),
      child: Card(
        color: Colors.white,
        elevation: 1.5,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: gradeColor.withValues(alpha: 0.35), width: 1.2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Thumbnail with badge
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (item.thumbnailBytes != null)
                    ClipRRect(
                      borderRadius: const BorderRadius.vertical(top: Radius.circular(11)),
                      child: Image.memory(item.thumbnailBytes!, fit: BoxFit.cover),
                    )
                  else
                    Container(
                      color: Colors.grey.shade100,
                      child: const Icon(Icons.image, color: Colors.grey),
                    ),
                  Positioned(
                    top: 6,
                    right: 6,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.75),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        '${(r.confidence * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          color: r.accepted ? const Color(0xFF00E676) : const Color(0xFFFF5252),
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 6,
                    left: 6,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1.5),
                      decoration: BoxDecoration(
                        color: gradeColor,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        r.label.split('_').first.toUpperCase(),
                        style: const TextStyle(color: Colors.white, fontSize: 8.5, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.fileName,
                    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'FP: ${r.imageFingerprint.substring(0, 6)}...',
                        style: TextStyle(fontSize: 9.5, color: Colors.grey.shade600, fontFamily: 'monospace'),
                      ),
                      Text(
                        '${r.totalMs}ms',
                        style: TextStyle(fontSize: 9.5, color: Colors.grey.shade600),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTableRow(BatchItem item) {
    if (item.result == null) {
      return Card(
        color: const Color(0xFFFFEBEE),
        margin: EdgeInsets.zero,
        child: ListTile(
          dense: true,
          leading: const Icon(Icons.error_outline, color: Colors.red),
          title: Text(item.fileName, style: const TextStyle(fontSize: 12)),
          subtitle: const Text('Failed to analyze', style: TextStyle(fontSize: 10, color: Colors.red)),
        ),
      );
    }

    final r = item.result!;
    final gradeColor = _getGradeColor(r.label);

    return InkWell(
      onTap: () => _showItemDetailModal(item),
      borderRadius: BorderRadius.circular(10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.grey.shade200),
        ),
        child: Row(
          children: [
            if (item.thumbnailBytes != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: Image.memory(item.thumbnailBytes!, width: 40, height: 40, fit: BoxFit.cover),
              )
            else
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(color: Colors.grey.shade200, borderRadius: BorderRadius.circular(6)),
                child: const Icon(Icons.image, size: 20, color: Colors.grey),
              ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(item.fileName, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12), maxLines: 1),
                  const SizedBox(height: 2),
                  Text('ID: ${r.imageFingerprint} | ${r.totalMs} ms', style: TextStyle(fontSize: 10, color: Colors.grey.shade600, fontFamily: 'monospace')),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(color: gradeColor.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(6)),
                  child: Text(
                    r.label.replaceAll('_', ' ').toUpperCase(),
                    style: TextStyle(fontSize: 9.5, fontWeight: FontWeight.bold, color: gradeColor),
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${(r.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: r.accepted ? const Color(0xFF2E7D32) : const Color(0xFFC62828),
                  ),
                ),
              ],
            ),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right, size: 16, color: Colors.grey),
          ],
        ),
      ),
    );
  }
}
