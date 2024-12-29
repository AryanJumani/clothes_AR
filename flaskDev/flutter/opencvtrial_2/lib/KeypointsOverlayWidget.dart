import 'package:flutter/material.dart';

import 'KeyPointsPainter.dart';

class KeypointsOverlayWidget extends StatelessWidget {
  final List<List<double>> keypoints;

  KeypointsOverlayWidget({required this.keypoints});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: KeypointsPainter(keypoints),
      child: Container(),
    );
  }
}