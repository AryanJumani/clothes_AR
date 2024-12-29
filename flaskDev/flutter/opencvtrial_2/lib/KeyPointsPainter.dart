import 'package:flutter/material.dart';

class KeypointsPainter extends CustomPainter {
  final List<List<double>> keypoints;

  KeypointsPainter(this.keypoints);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 4.0;

    for (var keypoint in keypoints) {
      if (keypoint[2] > 0.5) { // Confidence threshold
        final x = keypoint[0] * size.width; // Scale to screen dimensions
        final y = keypoint[1] * size.height;
        canvas.drawCircle(Offset(x, y), 5.0, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}