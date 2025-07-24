import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import '../utils/transition_utils.dart';

class VectorizedSvgScreen extends StatefulWidget {
  final String originalImagePath;
  final String croppedImagePath;
  final String svgString;

  const VectorizedSvgScreen({
    super.key,
    required this.originalImagePath,
    required this.croppedImagePath,
    required this.svgString,
  });

  @override
  _VectorizedSvgScreenState createState() => _VectorizedSvgScreenState();
}

class _VectorizedSvgScreenState extends State<VectorizedSvgScreen> {
  bool _isSaving = false;
  String? _savedFilePath;

  @override
  void initState() {
    super.initState();
    // Show success message
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Logo successfully vectorized!'),
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
        title: const Text('Vectorized Logo'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.popWithAnimation(
              transitionType: TransitionType.slideDown),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: _saveSvgFile,
            tooltip: 'Save SVG',
          ),
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _shareSvgFile,
            tooltip: 'Share SVG',
          ),
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () => _navigateToHome(context),
            tooltip: 'Go to Home',
          ),
        ],
      ),
      body: _isSaving
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Expanded(
                  child: _buildSvgView(),
                ),
                _buildComparisonView(),
                _buildInfoPanel(),
              ],
            ),
    );
  }

  Widget _buildSvgView() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.all(16.0),
      child: Center(
        child: SingleChildScrollView(
          child: Hero(
            tag: 'logo_image',
            child: SvgPicture.string(
              widget.svgString,
              width: 300,
              height: 300,
              fit: BoxFit.contain,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildComparisonView() {
    return Container(
      height: 150,
      color: Colors.grey[100],
      padding: const EdgeInsets.all(8.0),
      child: Row(
        children: [
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  'Original',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: Image.file(
                    File(widget.croppedImagePath),
                    fit: BoxFit.contain,
                  ),
                ),
              ],
            ),
          ),
          const VerticalDivider(width: 16, thickness: 1),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  'Vectorized',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: Container(
                    color: Colors.white,
                    child: SvgPicture.string(
                      widget.svgString,
                      fit: BoxFit.contain,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
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
            'Vectorized SVG Logo',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Your logo has been vectorized and can now be scaled to any size without losing quality.',
            style: TextStyle(fontSize: 16),
          ),
          const SizedBox(height: 16),
          if (_savedFilePath != null)
            Text(
              'Saved to: $_savedFilePath',
              style: const TextStyle(
                fontSize: 14,
                color: Colors.green,
              ),
            ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                icon: const Icon(Icons.save_alt),
                label: const Text('Save SVG'),
                onPressed: _saveSvgFile,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                ),
              ),
              ElevatedButton.icon(
                icon: const Icon(Icons.share),
                label: const Text('Share'),
                onPressed: _shareSvgFile,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _saveSvgFile() async {
    setState(() {
      _isSaving = true;
    });

    try {
      // First try to use Downloads directory, fall back to Documents if not available
      Directory? directory;
      try {
        directory = await getDownloadsDirectory();
      } catch (e) {
        // If getDownloadsDirectory throws an error, we'll use the app documents directory
      }

      // Fall back to app documents directory if Downloads not available
      directory ??= await getApplicationDocumentsDirectory();

      final fileName =
          'vectorized_logo_${DateTime.now().millisecondsSinceEpoch}.svg';
      final filePath = '${directory.path}/$fileName';

      // Write the SVG content to file
      final file = File(filePath);
      await file.writeAsString(widget.svgString);

      setState(() {
        _savedFilePath = filePath;
        _isSaving = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('SVG saved to: $filePath'),
            backgroundColor: Theme.of(context).colorScheme.tertiary,
            behavior: SnackBarBehavior.floating,
            action: SnackBarAction(
              label: 'OK',
              onPressed: () {},
              textColor: Theme.of(context).colorScheme.onTertiary,
            ),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _isSaving = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error saving SVG: ${e.toString()}'),
            backgroundColor: Theme.of(context).colorScheme.error,
            behavior: SnackBarBehavior.floating,
            action: SnackBarAction(
              label: 'Dismiss',
              onPressed: () {},
              textColor: Theme.of(context).colorScheme.onError,
            ),
          ),
        );
      }
    }
  }

  Future<void> _shareSvgFile() async {
    setState(() {
      _isSaving = true;
    });

    try {
      // Always create a temporary file for sharing
      // This ensures we have a fresh file and valid path
      final directory = await getApplicationDocumentsDirectory();
      final fileName =
          'vectorized_logo_${DateTime.now().millisecondsSinceEpoch}.svg';
      final filePath = '${directory.path}/$fileName';

      // Write the SVG content to file
      final file = File(filePath);
      await file.writeAsString(widget.svgString);

      // Update saved path if we don't have one yet
      if (_savedFilePath == null) {
        _savedFilePath = filePath;
      } // Share the file using XFile
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'My Vectorized Logo',
      );

      setState(() {
        _isSaving = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('SVG file shared successfully'),
            backgroundColor: Theme.of(context).colorScheme.tertiary,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } catch (e) {
      setState(() {
        _isSaving = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error sharing SVG: ${e.toString()}'),
            backgroundColor: Theme.of(context).colorScheme.error,
            behavior: SnackBarBehavior.floating,
            action: SnackBarAction(
              label: 'Dismiss',
              onPressed: () {},
              textColor: Theme.of(context).colorScheme.onError,
            ),
          ),
        );
      }
    }
  }

  void _navigateToHome(BuildContext context) {
    // Use the extension method for consistent home navigation
    context.navigateToHome();
  }
}
