import 'package:flutter/material.dart';
import '../models/detection_model.dart';
import '../utils/transition_utils.dart';
import 'logo_detection_screen.dart';

class ResultScreen extends StatefulWidget {
  final String imagePath;
  final List<Detection> detections;

  const ResultScreen({
    super.key,
    required this.imagePath,
    required this.detections,
  });

  @override
  _ResultScreenState createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  String? _finalImagePath;

  @override
  void initState() {
    super.initState();
    // Immediately navigate to the LogoDetectionScreen
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _navigateToLogoDetection();
    });
  }

  void _navigateToLogoDetection() async {
    final result = await context.pushSmoothRoute(
      page: LogoDetectionScreen(
        imagePath: widget.imagePath,
        detections: widget.detections,
      ),
      transitionType: TransitionType.fade,
      routeName: 'logo_detection',
    );

    // If we got a result back, set it as the final image path
    if (result != null && result is String) {
      setState(() {
        _finalImagePath = result;
      });
      // Return this result to the previous screen
      Navigator.pop(context, _finalImagePath);
    } else {
      // If the user cancelled the flow, just go back
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    // This is just a loading screen while we transition to the LogoDetectionScreen
    return const Scaffold(
      body: Center(
        child: CircularProgressIndicator(),
      ),
    );
  }
}
