import 'dart:io';
import 'package:flutter/material.dart';
import '../models/detection_model.dart';
import '../services/vectorize_service.dart';
import '../utils/transition_utils.dart';
import 'vectorized_svg_screen.dart';

class CroppedImageScreen extends StatefulWidget {
  final String originalImagePath;
  final String croppedImagePath;
  final BoundingBox boundingBox;

  const CroppedImageScreen({
    super.key,
    required this.originalImagePath,
    required this.croppedImagePath,
    required this.boundingBox,
  });

  @override
  _CroppedImageScreenState createState() => _CroppedImageScreenState();
}

class _CroppedImageScreenState extends State<CroppedImageScreen> {
  bool _isLoading = false;
  bool _isVectorizing = false;
  String? _currentCroppedImagePath;
  @override
  void initState() {
    super.initState();
    _currentCroppedImagePath = widget.croppedImagePath;

    // Show success message
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content:
              Text('Image cropped successfully! You can adjust if needed.'),
          backgroundColor: Colors.green,
          duration: Duration(seconds: 2),
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cropped Logo'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.edit),
            onPressed: _goBackToEdit,
            tooltip: 'Edit Crop',
          ),
          IconButton(
            icon: const Icon(Icons.check),
            onPressed: _saveAndProceed,
            tooltip: 'Proceed',
          ),
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () => _navigateToHome(context),
            tooltip: 'Go to Home',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: _buildCroppedImageView(),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16.0),
                  child: ElevatedButton.icon(
                    icon: _isVectorizing
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.auto_awesome),
                    label: const Text('Vectorize Logo'),
                    onPressed: _isVectorizing ? null : _vectorizeImage,
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 12,
                      ),
                    ),
                  ),
                ),
                _buildInfoPanel(),
              ],
            ),
    );
  }

  Widget _buildCroppedImageView() {
    return Stack(
      fit: StackFit.expand,
      children: [
        if (_currentCroppedImagePath != null)
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Hero(
              tag: 'logo_image',
              child: Image.file(
                File(_currentCroppedImagePath!),
                fit: BoxFit.contain,
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildInfoPanel() {
    return Container(
      padding: const EdgeInsets.all(16),
      color: Colors.grey[200],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'Cropped Logo Preview',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'You can edit the crop if needed, or proceed to vectorize the logo.',
            style: TextStyle(fontSize: 16),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              OutlinedButton.icon(
                icon: const Icon(Icons.edit),
                label: const Text('Edit Selection'),
                onPressed: _goBackToEdit,
              ),
              const SizedBox(width: 8),
              ElevatedButton.icon(
                icon: const Icon(Icons.save),
                label: const Text('Save Image'),
                onPressed: _saveAndProceed,
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _goBackToEdit() {
    context.popWithAnimation(transitionType: TransitionType.slideRight);
  }

  void _saveAndProceed() {
    if (_currentCroppedImagePath == null) return;

    // Return to previous screen with the cropped image path
    context.popWithAnimation(
      transitionType: TransitionType.slideRight,
      result: _currentCroppedImagePath,
    );
  }

  void _navigateToHome(BuildContext context) {
    // Use the extension method for consistent home navigation
    context.navigateToHome();
  }

  Future<void> _vectorizeImage() async {
    if (_currentCroppedImagePath == null) return;

    setState(() {
      _isVectorizing = true;
    });

    try {
      final file = File(_currentCroppedImagePath!);
      final svgString = await VectorizeService.vectorizeImage(file);

      if (mounted) {
        setState(() {
          _isVectorizing = false;
        }); // Navigate to vectorized SVG screen with a smooth transition
        context.pushSmoothRoute(
          page: VectorizedSvgScreen(
            originalImagePath: widget.originalImagePath,
            croppedImagePath: _currentCroppedImagePath!,
            svgString: svgString,
          ),
          transitionType: TransitionType.slideUp,
          routeName: 'vectorized_svg',
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isVectorizing = false;
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}
