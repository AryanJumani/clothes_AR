import 'dart:io';

import 'package:clothes_trial_3/Segregator.dart';
import 'package:clothes_trial_3/constants.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class UploadPhoto extends StatefulWidget {
  const UploadPhoto({super.key});

  @override
  State<UploadPhoto> createState() => _UploadPhotoState();
}

class _UploadPhotoState extends State<UploadPhoto> {
  File? _imageFile;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final XFile? picked = await _picker.pickImage(source: source);
    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Pick an Image')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _imageFile != null
                ? Image.file(_imageFile!, height: 300)
                : Text('No image selected'),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _pickImage(ImageSource.gallery),
              child: Text('Pick from Gallery'),
            ),
            IconButton(
              onPressed: () {
                if (_imageFile != null) {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder:
                          (context) => Segregator(imagePath: _imageFile!.path),
                    ),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Please pick an image first')),
                  );
                }
              },
              icon: Icon(Icons.arrow_circle_right, size: 100, color: free),
              color: salmon,
            ),
          ],
        ),
      ),
      backgroundColor: background,
    );
  }
}
