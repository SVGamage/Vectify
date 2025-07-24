import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/logo_detection_service.dart';
import '../models/detection_model.dart';
import 'result_screen.dart';
import '../utils/image_utils.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  bool _isLoading = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Vectify',
          style: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.bold,
            color: Colors.blue,
          ),
          textAlign: TextAlign.center,
        ),
        centerTitle: true,
      ),
      body: Container(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 30),
            const Text(
              'Tap a button below to capture or select a photo',
              style: TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              'Processing will start automatically',
              style: TextStyle(
                  fontSize: 14,
                  fontStyle: FontStyle.italic,
                  color: Colors.grey),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 40),

            // Preview of selected image
            if (_selectedImage != null)
              Expanded(
                child: Container(
                  margin: const EdgeInsets.only(bottom: 20),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image.file(
                      _selectedImage!,
                      fit: BoxFit.contain,
                    ),
                  ),
                ),
              )
            else
              Expanded(
                child: Container(
                  margin: const EdgeInsets.only(bottom: 20),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Center(
                    child: Icon(
                      Icons.image,
                      size: 100,
                      color: Colors.grey,
                    ),
                  ),
                ),
              ), // Buttons for camera and gallery or loading indicator
            _isLoading
                ? const Center(
                    child: Column(
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Processing image...',
                            style: TextStyle(fontSize: 16)),
                      ],
                    ),
                  )
                : Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isLoading ? null : _captureImage,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                      const SizedBox(width: 20),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isLoading ? null : _pickImage,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                    ],
                  ),

            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Future<void> _captureImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    _processPickedImage(image);
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    _processPickedImage(image);
  }

  void _processPickedImage(XFile? image) async {
    if (image != null) {
      // Set the selected image
      setState(() {
        _selectedImage = File(image.path);
      });

      // Automatically process the image
      await _processImage();
    }
  }

  Future<void> _processImage() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      final response = await LogoDetectionService.detectLogo(_selectedImage!);

      if (!mounted) return;

      setState(() {
        _isLoading = false;
      });
      if (response.success && response.detections.isNotEmpty) {
        // Navigate to result screen
        final croppedImagePath = await Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultScreen(
              imagePath: _selectedImage!.path,
              detections: response.detections,
            ),
          ),
        );

        // Show the cropped image if available
        if (croppedImagePath != null && croppedImagePath is String) {
          if (mounted) {
            _showCroppedImage(croppedImagePath);
          }
        }
      } else {
        // No logo detected - show manual selection dialog
        if (response.success) {
          _showNoLogoDetectedDialog();
        } else {
          _showManualSelectionDialog();
        }
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showErrorDialog('Error: $e');
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(message),
            const SizedBox(height: 16),
            const Text(
              'Would you like to manually select a logo area instead?',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _navigateToManualSelection();
            },
            child: const Text('Select Manually'),
          ),
        ],
      ),
    );
  }

  void _showManualSelectionDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('No Logo Detected'),
        content: const Text(
            'No logos were automatically detected in the image. Would you like to manually select a logo area?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _navigateToManualSelection();
            },
            child: const Text('Select Manually'),
          ),
        ],
      ),
    );
  }

  void _showNoLogoDetectedDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('No Logo Detected'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('No logos were automatically detected in the image.'),
            const SizedBox(height: 16),
            const Text('Possible reasons:'),
            const SizedBox(height: 8),
            const Text('• The logo is too small or low contrast'),
            const Text('• The image quality is not sufficient'),
            const Text('• The logo design is not recognizable by our system'),
            const SizedBox(height: 16),
            const Text('Would you like to manually select a logo area?'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _navigateToManualSelection();
            },
            child: const Text('Select Manually'),
          ),
        ],
      ),
    );
  }

  void _navigateToManualSelection() async {
    if (_selectedImage == null) return;

    try {
      // Use the utility method to create a default bounding box
      BoundingBox defaultBox =
          await ImageUtils.createDefaultBoundingBox(_selectedImage!.path);

      // Create a single detection with the default box
      Detection manualDetection = Detection(
        boundingBox: defaultBox,
        classId: 0,
        className: 'Manual Selection',
        confidence: 1.0,
      );

      // Navigate to result screen with the manual detection
      final croppedImagePath = await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultScreen(
            imagePath: _selectedImage!.path,
            detections: [manualDetection],
          ),
        ),
      );

      // Show the cropped image if available
      if (croppedImagePath != null && croppedImagePath is String && mounted) {
        _showCroppedImage(croppedImagePath);
      }
    } catch (e) {
      _showErrorDialog('Error preparing manual selection: $e');
    }
  }

  void _showCroppedImage(String imagePath) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AppBar(
              title: const Text('Cropped Logo'),
              automaticallyImplyLeading: false,
              actions: [
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Image.file(
                File(imagePath),
                fit: BoxFit.contain,
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(
                'Logo cropped successfully!',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 16.0),
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pop(context);
                },
                child: const Text('Done'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
