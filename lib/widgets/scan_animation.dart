import 'dart:math' as math;
import 'package:flutter/material.dart';

/// Custom Shimmer effect without external dependencies.
/// Uses an [AnimationController] with a sliding LinearGradient.
class ShimmerEffect extends StatefulWidget {
  final Widget child;
  final Color baseColor;
  final Color highlightColor;
  final Duration duration;

  const ShimmerEffect({
    super.key,
    required this.child,
    this.baseColor = const Color(0xFFE0E0E0),
    this.highlightColor = const Color(0xFFF5F5F5),
    this.duration = const Duration(milliseconds: 1400),
  });

  @override
  State<ShimmerEffect> createState() => _ShimmerEffectState();
}

class _ShimmerEffectState extends State<ShimmerEffect>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration)
      ..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return ShaderMask(
          blendMode: BlendMode.srcATop,
          shaderCallback: (bounds) {
            final double value = _controller.value;
            // Slide gradient from -1.0 to 2.0 to cover bounds smoothly
            final double dx = (bounds.width * 2) * value - bounds.width;
            return LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                widget.baseColor,
                widget.highlightColor,
                widget.baseColor,
              ],
              stops: const [0.1, 0.5, 0.9],
              transform: _SlidingGradientTransform(slidePercent: dx, totalWidth: bounds.width),
            ).createShader(bounds);
          },
          child: widget.child,
        );
      },
      child: widget.child,
    );
  }
}

class _SlidingGradientTransform extends GradientTransform {
  final double slidePercent;
  final double totalWidth;

  const _SlidingGradientTransform({
    required this.slidePercent,
    required this.totalWidth,
  });

  @override
  Matrix4? transform(Rect bounds, {TextDirection? textDirection}) {
    return Matrix4.translationValues(slidePercent, 0.0, 0.0);
  }
}

/// A placeholder box for skeleton loading.
class ShimmerBox extends StatelessWidget {
  final double width;
  final double height;
  final double borderRadius;
  final Color? color;

  const ShimmerBox({
    super.key,
    required this.width,
    required this.height,
    this.borderRadius = 8,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: color ?? const Color(0xFFE8EAE6),
        borderRadius: BorderRadius.circular(borderRadius),
      ),
    );
  }
}

/// Shimmer Skeleton that mirrors the actual Cinnamon Edge Result Card
class ShimmerResultCard extends StatelessWidget {
  const ShimmerResultCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.white,
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: ShimmerEffect(
          baseColor: const Color(0xFFE8EBE7),
          highlightColor: const Color(0xFFFAFAFA),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header pill
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const ShimmerBox(width: 140, height: 16, borderRadius: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE8EBE7),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const ShimmerBox(width: 50, height: 14, borderRadius: 6),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Grade title and confidence badge skeleton
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        ShimmerBox(width: 100, height: 12, borderRadius: 4),
                        SizedBox(height: 6),
                        ShimmerBox(width: 180, height: 26, borderRadius: 6),
                      ],
                    ),
                  ),
                  Container(
                    width: 76,
                    height: 52,
                    decoration: BoxDecoration(
                      color: const Color(0xFFE8EBE7),
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),

              // Gate status skeleton
              const ShimmerBox(width: 220, height: 14, borderRadius: 4),
              const SizedBox(height: 16),
              const Divider(height: 1, color: Color(0xFFEEEEEE)),
              const SizedBox(height: 16),

              // 3 Timings skeleton cards
              const Row(
                children: [
                  Expanded(child: ShimmerBox(width: double.infinity, height: 48, borderRadius: 8)),
                  SizedBox(width: 8),
                  Expanded(child: ShimmerBox(width: double.infinity, height: 48, borderRadius: 8)),
                  SizedBox(width: 8),
                  Expanded(child: ShimmerBox(width: double.infinity, height: 48, borderRadius: 8)),
                ],
              ),
              const SizedBox(height: 18),
              const Divider(height: 1, color: Color(0xFFEEEEEE)),
              const SizedBox(height: 16),

              // Class probabilities skeleton
              const ShimmerBox(width: 160, height: 14, borderRadius: 4),
              const SizedBox(height: 12),

              ...List.generate(4, (index) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        ShimmerBox(width: 110 + (index * 15.0), height: 12, borderRadius: 4),
                        const ShimmerBox(width: 36, height: 12, borderRadius: 4),
                      ],
                    ),
                    const SizedBox(height: 6),
                    const ShimmerBox(width: double.infinity, height: 6, borderRadius: 4),
                  ],
                ),
              )),
            ],
          ),
        ),
      ),
    );
  }
}

/// High-Tech AI Laser Scanner Overlay that renders on top of the image preview
class AiLaserScanner extends StatefulWidget {
  final String statusText;
  final double progressPercent;

  const AiLaserScanner({
    super.key,
    required this.statusText,
    required this.progressPercent,
  });

  @override
  State<AiLaserScanner> createState() => _AiLaserScannerState();
}

class _AiLaserScannerState extends State<AiLaserScanner>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Stack(
          children: [
            // Dark vignette overlay with subtle cyber grid
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      Colors.black.withValues(alpha: 0.35),
                      Colors.black.withValues(alpha: 0.15),
                      Colors.black.withValues(alpha: 0.55),
                    ],
                  ),
                ),
              ),
            ),

            // Custom Painted HUD: Reticle corners & Laser scanning beam
            Positioned.fill(
              child: CustomPaint(
                painter: _LaserBeamPainter(
                  beamPosition: _controller.value,
                  primaryColor: const Color(0xFF00E676), // Vivid Emerald AI Beam
                  secondaryColor: const Color(0xFF1B4D3E),
                ),
              ),
            ),

            // Top AI Status Tag
            Positioned(
              top: 12,
              left: 12,
              right: 12,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.65),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: const Color(0xFF00E676).withValues(alpha: 0.6),
                        width: 1,
                      ),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _PulsingDot(color: Color(0xFF00E676)),
                        SizedBox(width: 6),
                        Text(
                          'AI SCAN ACTIVE',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 10.5,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.8,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.65),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${(widget.progressPercent * 100).toInt()}%',
                      style: const TextStyle(
                        color: Color(0xFF00E676),
                        fontWeight: FontWeight.bold,
                        fontSize: 11,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Bottom Scanning Status Badge with Glassmorphism
            Positioned(
              bottom: 12,
              left: 12,
              right: 12,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  decoration: BoxDecoration(
                    color: const Color(0xFF112211).withValues(alpha: 0.85),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: const Color(0xFF00E676).withValues(alpha: 0.35),
                      width: 1,
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF00E676)),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              widget.statusText,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 11.5,
                                fontWeight: FontWeight.w600,
                              ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      // Smooth Progress Line
                      ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: LinearProgressIndicator(
                          value: widget.progressPercent,
                          minHeight: 3,
                          backgroundColor: Colors.white.withValues(alpha: 0.15),
                          valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF00E676)),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _PulsingDot extends StatefulWidget {
  final Color color;
  const _PulsingDot({required this.color});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse;

  @override
  void initState() {
    super.initState();
    _pulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _pulse,
      builder: (context, child) {
        return Container(
          width: 7,
          height: 7,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: widget.color.withValues(alpha: 0.5 + (_pulse.value * 0.5)),
            boxShadow: [
              BoxShadow(
                color: widget.color.withValues(alpha: 0.8 * _pulse.value),
                blurRadius: 6 * _pulse.value,
                spreadRadius: 2 * _pulse.value,
              ),
            ],
          ),
        );
      },
    );
  }
}

/// Custom painter for the laser beam and cyber corner reticles
class _LaserBeamPainter extends CustomPainter {
  final double beamPosition; // 0.0 to 1.0 (top to bottom)
  final Color primaryColor;
  final Color secondaryColor;

  _LaserBeamPainter({
    required this.beamPosition,
    required this.primaryColor,
    required this.secondaryColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double margin = 14.0;
    final double cornerLen = 22.0;
    final double cornerThick = 2.5;

    final cornerPaint = Paint()
      ..color = primaryColor.withValues(alpha: 0.85)
      ..strokeWidth = cornerThick
      ..style = PaintingStyle.stroke;

    // Draw 4 corner targeting reticles: [ ]
    // Top-Left
    canvas.drawLine(Offset(margin, margin), Offset(margin + cornerLen, margin), cornerPaint);
    canvas.drawLine(Offset(margin, margin), Offset(margin, margin + cornerLen), cornerPaint);

    // Top-Right
    canvas.drawLine(Offset(size.width - margin, margin), Offset(size.width - margin - cornerLen, margin), cornerPaint);
    canvas.drawLine(Offset(size.width - margin, margin), Offset(size.width - margin, margin + cornerLen), cornerPaint);

    // Bottom-Left
    canvas.drawLine(Offset(margin, size.height - margin), Offset(margin + cornerLen, size.height - margin), cornerPaint);
    canvas.drawLine(Offset(margin, size.height - margin), Offset(margin, size.height - margin - cornerLen), cornerPaint);

    // Bottom-Right
    canvas.drawLine(Offset(size.width - margin, size.height - margin), Offset(size.width - margin - cornerLen, size.height - margin), cornerPaint);
    canvas.drawLine(Offset(size.width - margin, size.height - margin), Offset(size.width - margin, size.height - margin - cornerLen), cornerPaint);

    // Calculate vertical position of laser beam
    final double currentY = margin + (size.height - 2 * margin) * beamPosition;

    // Laser glow gradient (a soft light curtain following the beam)
    final double glowHeight = 38.0;
    final Rect glowRect = Rect.fromLTRB(
      margin + 2,
      math.max(margin, currentY - glowHeight),
      size.width - margin - 2,
      math.min(size.height - margin, currentY + glowHeight),
    );

    final glowPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          primaryColor.withValues(alpha: 0.0),
          primaryColor.withValues(alpha: 0.22),
          primaryColor.withValues(alpha: 0.0),
        ],
      ).createShader(glowRect);

    canvas.drawRect(glowRect, glowPaint);

    // Main intense laser line
    final laserPaint = Paint()
      ..shader = LinearGradient(
        colors: [
          primaryColor.withValues(alpha: 0.1),
          Colors.white,
          primaryColor,
          Colors.white,
          primaryColor.withValues(alpha: 0.1),
        ],
        stops: const [0.0, 0.25, 0.5, 0.75, 1.0],
      ).createShader(Rect.fromLTWH(margin, currentY - 1, size.width - 2 * margin, 2))
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(
      Offset(margin + 2, currentY),
      Offset(size.width - margin - 2, currentY),
      laserPaint,
    );

    // Laser bloom points at left and right edges
    final dotPaint = Paint()..color = Colors.white;
    canvas.drawCircle(Offset(margin + 2, currentY), 2.5, dotPaint);
    canvas.drawCircle(Offset(size.width - margin - 2, currentY), 2.5, dotPaint);
  }

  @override
  bool shouldRepaint(covariant _LaserBeamPainter oldDelegate) {
    return oldDelegate.beamPosition != beamPosition ||
        oldDelegate.primaryColor != primaryColor;
  }
}
