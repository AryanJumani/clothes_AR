package com.example.opencvtrial_2

import android.util.Base64
import android.graphics.BitmapFactory
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.android.FlutterActivity
import io.flutter.plugin.common.MethodChannel
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.core.Mat
import org.opencv.android.Utils
import org.opencv.imgproc.Imgproc
import org.opencv.core.Size
import org.opencv.core.Scalar

class MainActivity : FlutterActivity() {

    private lateinit var net: Net

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Load the ONNX model
        val modelPath = applicationContext.getExternalFilesDir(null)!!.absolutePath + "/pose_landmark_full.onnx"
        net = Dnn.readNetFromONNX(modelPath)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "torsodetection")
            .setMethodCallHandler { call, result ->
                if (call.method == "detectTorso") {
                    val imageBase64 = call.argument<String>("image")
                    if (imageBase64 != null) {
                        val imageBytes = Base64.decode(imageBase64, Base64.DEFAULT)
                        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                        val keypoints = detectTorso(bitmap)
                        result.success(keypoints)
                    } else {
                        result.error("INVALID_ARGUMENT", "Missing image data", null)
                    }
                } else {
                    result.notImplemented()
                }
            }
    }

    private fun detectTorso(bitmap: android.graphics.Bitmap): List<List<Float>> {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)

        // Preprocess the image for the model
        val blob = Dnn.blobFromImage(mat, 1.0 / 255.0, Size(256.0, 256.0), Scalar(0.0, 0.0, 0.0), true, false)
        net.setInput(blob)

        // Perform inference
        val output = net.forward()

        // Extract keypoints (x, y, confidence)
        val keypoints = mutableListOf<List<Float>>()
        for (i in 0 until output.rows()) {
            val x = output[i, 0][0]
            val y = output[i, 1][0]
            val confidence = output[i, 2][0]
            keypoints.add(listOf(x.toFloat(), y.toFloat(), confidence.toFloat()))
        }

        return keypoints
    }
}