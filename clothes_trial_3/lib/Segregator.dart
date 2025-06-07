import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:clothes_trial_3/webview.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'constants.dart';

class Segregator extends StatefulWidget {
  final String imagePath;

  const Segregator({super.key, required this.imagePath});

  @override
  State<Segregator> createState() => _SegregatorState();
}

class _SegregatorState extends State<Segregator> with TickerProviderStateMixin {
  Uint8List? segmentedImage;
  bool isLoading = true;
  bool showOverlay = false;
  late AnimationController _revealController;
  ui.Image? gridImage;

  @override
  void initState() {
    super.initState();
    _revealController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        setState(() => showOverlay = true);
      }
    });

    _loadGridImage();
    _uploadAndSegmentImage();
  }

  Future<void> _loadGridImage() async {
    final data = await rootBundle.load('assets/grid.png');
    final codec = await ui.instantiateImageCodec(data.buffer.asUint8List());
    final frame = await codec.getNextFrame();
    setState(() {
      gridImage = frame.image;
    });
  }

  Future<void> _uploadAndSegmentImage() async {
    final uri = Uri.parse("$baseUrl/segment");
    final imageFile = File(widget.imagePath);
    final mimeType =
        lookupMimeType(imageFile.path)?.split('/') ?? ['image', 'png'];

    final request = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          contentType: MediaType(mimeType[0], mimeType[1]),
        ),
      );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      setState(() {
        segmentedImage = response.bodyBytes;
        isLoading = false;
      });
      _revealController.forward();
    } else {
      setState(() => isLoading = false);
      print('Segmentation failed: ${response.statusCode}');
    }
  }

  @override
  void dispose() {
    _revealController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final imageWidth = screenSize.width * 0.8;

    if (isLoading || gridImage == null) {
      return const Scaffold(
        backgroundColor: background,
        body: Center(child: CircularProgressIndicator(color: aqua)),
      );
    }

    return Scaffold(
      backgroundColor: background,
      body: Stack(
        children: [
          // Centered original image
          Center(
            child: Image.file(
              File(widget.imagePath),
              width: imageWidth,
              fit: BoxFit.contain,
            ),
          ),

          // Radial reveal of REPEATED grid over full screen
          if (!showOverlay)
            Positioned.fill(
              child: AnimatedBuilder(
                animation: _revealController,
                builder: (context, _) {
                  return ClipPath(
                    clipper: RadialRevealClipper(_revealController.value),
                    child: CustomPaint(
                      painter: RepeatedGridPainter(gridImage!),
                    ),
                  );
                },
              ),
            ),

          // Fullscreen blur overlay
          if (showOverlay)
            Positioned.fill(
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                child: Container(color: Colors.transparent),
              ),
            ),

          // Centered segmented image
          if (showOverlay && segmentedImage != null)
            Center(
              child: Image.memory(
                segmentedImage!,
                width: imageWidth,
                fit: BoxFit.contain,
              ),
            ),

          // Buttons
          if (showOverlay)
            Positioned(
              bottom: 30,
              left: 0,
              right: 0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: Icon(Icons.arrow_back),
                  ),
                  const SizedBox(width: 20),
                  IconButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder:
                              (_) => WebView(segmentedImage: segmentedImage!),
                        ),
                      );
                    },
                    icon: Icon(Icons.check_outlined),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class RadialRevealClipper extends CustomClipper<Path> {
  final double progress;

  RadialRevealClipper(this.progress);

  @override
  Path getClip(Size size) {
    final radius = size.longestSide * progress;
    return Path()..addOval(
      Rect.fromCircle(center: size.center(Offset.zero), radius: radius),
    );
  }

  @override
  bool shouldReclip(RadialRevealClipper oldClipper) {
    return oldClipper.progress != progress;
  }
}

class RepeatedGridPainter extends CustomPainter {
  final ui.Image gridImage;

  RepeatedGridPainter(this.gridImage);

  @override
  void paint(Canvas canvas, Size size) {
    final paint =
        Paint()
          ..shader = ImageShader(
            gridImage,
            TileMode.repeated,
            TileMode.repeated,
            Matrix4.identity().scaled(1.0).storage,
          );

    canvas.drawRect(Offset.zero & size, paint);
  }

  @override
  bool shouldRepaint(covariant RepeatedGridPainter oldDelegate) {
    return oldDelegate.gridImage != gridImage;
  }
}
