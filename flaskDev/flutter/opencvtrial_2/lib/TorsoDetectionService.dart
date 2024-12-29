import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';

class TorsoDetectionService {
  static const platform = MethodChannel('torsodetection');

  static Future<List<List<double>>> detectTorso(Uint8List imageBytes) async {
    try {
      final base64Image = base64Encode(imageBytes);
      final result = await platform.invokeMethod('detectTorso', {"image": base64Image});
      return List<List<double>>.from(result.map((r) => List<double>.from(r)));
    } catch (e) {
      print("Error detecting torso: $e");
      return [];
    }
  }
}