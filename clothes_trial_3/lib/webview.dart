import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:permission_handler/permission_handler.dart';

class WebView extends StatefulWidget {
  final Uint8List segmentedImage;

  const WebView({super.key, required this.segmentedImage});

  @override
  State<WebView> createState() => _WebViewState();
}

class _WebViewState extends State<WebView> {
  InAppWebViewController? webViewController;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    await Permission.camera.request();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: InAppWebView(
        initialFile: 'assets/web/index.html',
        // adjust if in a different path
        initialSettings: InAppWebViewSettings(
          javaScriptEnabled: true,
          mediaPlaybackRequiresUserGesture: false,
          allowsInlineMediaPlayback: true,
          allowsBackForwardNavigationGestures: true,
          iframeAllow: "camera; microphone",
          iframeAllowFullscreen: true,
        ),
        onWebViewCreated: (controller) {
          webViewController = controller;
        },
        onLoadStop: (controller, url) async {
          // Convert image to base64 and inject it into the JS context
          final base64 = base64Encode(widget.segmentedImage);
          final js = """
            const img = new Image();
            img.src = "data:image/png;base64,$base64";
            img.onload = () => { window.shirtImg = img; };
          """;
          await controller.evaluateJavascript(source: js);
        },
        onPermissionRequest: (controller, request) async {
          return PermissionResponse(
            action: PermissionResponseAction.GRANT,
            resources: request.resources,
          );
        },
      ),
    );
  }
}
