import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:convert';
import 'package:flutter/services.dart';

import 'KeyPointsPainter.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Torso Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: TorsoDetectionScreen(),
    );
  }
}

class TorsoDetectionScreen extends StatefulWidget {
  @override
  _TorsoDetectionScreenState createState() => _TorsoDetectionScreenState();
}

class _TorsoDetectionScreenState extends State<TorsoDetectionScreen> {
  CameraController? _cameraController;
  List<List<double>> _keypoints = []; // Holds detected keypoints

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(cameras[0], ResolutionPreset.high);
    await _cameraController?.initialize();
    setState(() {});
  }

  Future<void> _processFrame() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    final image = await _cameraController!.takePicture();
    final imageBytes = await image.readAsBytes();

    try {
      final base64Image = base64Encode(imageBytes);
      final result = await MethodChannel('torsodetection')
          .invokeMethod<List<dynamic>>('detectTorso', {"image": base64Image});
      setState(() {
        _keypoints = result!
            .map((kp) => List<double>.from(kp))
            .toList(); // Convert keypoints to a list of doubles
      });
    } catch (e) {
      print("Error detecting torso: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Torso Detection')),
      body: Stack(
        children: [
          if (_cameraController != null && _cameraController!.value.isInitialized)
            CameraPreview(_cameraController!),
          if (_keypoints.isNotEmpty)
            CustomPaint(
              painter: KeypointsPainter(_keypoints),
              child: Container(),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _processFrame,
        child: Icon(Icons.camera),
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }
}