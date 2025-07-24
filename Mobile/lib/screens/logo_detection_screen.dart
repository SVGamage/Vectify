// filepath: d:\vectify-flutter-app\vectify_flutter_app\lib\screens\logo_detection_screen.dart
import 'package:flutter/material.dart';
import '../models/detection_model.dart';
import '../widgets/bounding_box_selector.dart';
import '../utils/image_utils.dart';
import '../utils/transition_utils.dart';
import 'cropped_image_screen.dart';

class LogoDetectionScreen extends StatefulWidget {
  final String imagePath;
  final List<Detection> detections;

  const LogoDetectionScreen({
    super.key,
    required this.imagePath,
    required this.detections,
  });

  @override
  _LogoDetectionScreenState createState() => _LogoDetectionScreenState();
}

class _LogoDetectionScreenState extends State<LogoDetectionScreen> {
  late BoundingBox _currentBoundingBox;
  bool _isLoading = false;
  bool _isManualSelection = false;

  @override
  void initState() {
    super.initState();
    // Initialize with the first detection's bounding box
    if (widget.detections.isNotEmpty) {
      _currentBoundingBox = widget.detections.first.boundingBox;

      // If this is a manual selection, set the flag
      if (widget.detections.first.className == 'Manual Selection') {
        _isManualSelection = true;

        // Show help dialog after widget is built
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _showInitialHelp();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          _isManualSelection
              ? 'Manual Logo Selection'
              : 'Logo Detection Result',
        ),
        actions: [
          if (_isManualSelection)
            Tooltip(
              message: 'Adjust the red box to select your logo',
              child: IconButton(
                icon: const Icon(Icons.help_outline),
                onPressed: _showHelpDialog,
              ),
            ),
          IconButton(
            icon: const Icon(Icons.crop),
            onPressed: _cropImage,
            tooltip: 'Crop and Continue',
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
                  child: Hero(
                    tag: 'logo_image',
                    child: BoundingBoxSelector(
                      imagePath: widget.imagePath,
                      detections: widget.detections,
                      initialEditingMode: _isManualSelection,
                      onBoundingBoxChanged: (box) {
                        setState(() {
                          _currentBoundingBox = box;
                        });
                      },
                    ),
                  ),
                ),
                _buildInfoPanel(),
              ],
            ),
    );
  }

  Widget _buildInfoPanel() {
    final detection = widget.detections.first;
    final bool isManual = detection.className == 'Manual Selection';

    return Container(
      padding: const EdgeInsets.all(16),
      color: Colors.grey[200],
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            isManual
                ? 'Manual Selection Mode'
                : 'Detected: ${detection.className}',
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          if (isManual)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Position the red box around the logo',
                  style: TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Icon(Icons.edit, size: 16, color: Colors.blue[700]),
                    const SizedBox(width: 4),
                    const Text('Use the Edit Box button to make adjustments'),
                  ],
                ),
                Row(
                  children: [
                    Icon(Icons.crop, size: 16, color: Colors.blue[700]),
                    const SizedBox(width: 4),
                    const Text('Tap Crop when done to continue'),
                  ],
                ),
              ],
            ),
        ],
      ),
    );
  }

  Future<void> _cropImage() async {
    if (_isLoading) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // First ensure the bounding box is properly normalized
      final normalizedBox = await ImageUtils.convertToNormalizedBoundingBox(
        widget.imagePath,
        _currentBoundingBox,
      );

      // Update the current bounding box to the normalized one
      setState(() {
        _currentBoundingBox = normalizedBox;
      });

      // Automatically crop the image using the normalized bounding box
      final croppedPath = await ImageUtils.cropImageWithBoundingBox(
        widget.imagePath,
        normalizedBox,
      );

      if (mounted) {
        setState(() {
          _isLoading = false;
        });

        // Navigate to the cropped image screen with a smooth transition
        context.pushSmoothRoute(
          page: CroppedImageScreen(
            originalImagePath: widget.imagePath,
            croppedImagePath: croppedPath,
            boundingBox: normalizedBox,
          ),
          transitionType: TransitionType.slideRight,
          routeName: 'cropped_image',
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to crop image: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _showHelpDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Manual Selection Help'),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'How to Select a Logo:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text('1. The red box shows your current selection'),
            Text('2. Tap "Edit Box" to enable editing mode'),
            Text('3. Drag the corners to resize the selection'),
            Text('4. Drag inside the box to move the entire selection'),
            Text('5. When finished, tap "Done" in the box'),
            Text('6. Tap the crop icon in the app bar to continue'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _navigateToHome(BuildContext context) {
    // Use the extension method for consistent home navigation
    context.navigateToHome();
  }

  void _showInitialHelp() {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Text('Adjust the red box to select your logo area'),
        duration: const Duration(seconds: 4),
        action: SnackBarAction(
          label: 'Help',
          onPressed: _showHelpDialog,
        ),
      ),
    );
  }
}
