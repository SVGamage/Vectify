import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'dart:ui' as ui;
import '../models/detection_model.dart';
import 'package:path_provider/path_provider.dart';

class ImageUtils {
  /// Convert from raw pixel coordinates to normalized 0-1000 range coordinates
  static Future<BoundingBox> convertToNormalizedBoundingBox(
    String imagePath,
    BoundingBox originalBox,
  ) async {
    // Get original image dimensions
    final File imageFile = File(imagePath);
    final image = await _getImageDimensions(imageFile);
    if (image == null) {
      // If we can't get dimensions, return the original
      return originalBox;
    }

    // Check if coordinates are already in 0-1000 range
    if (originalBox.x2 <= 1000 && originalBox.y2 <= 1000) {
      // Already normalized, but let's make sure the box is valid
      return BoundingBox(
        x1: originalBox.x1.clamp(0, 1000),
        y1: originalBox.y1.clamp(0, 1000),
        x2: originalBox.x2.clamp(0, 1000),
        y2: originalBox.y2.clamp(0, 1000),
      );
    }

    // Convert from pixel coordinates to 0-1000 range
    int normalizedX1 =
        (originalBox.x1 * 1000 / image.width).round().clamp(0, 1000);
    int normalizedY1 =
        (originalBox.y1 * 1000 / image.height).round().clamp(0, 1000);
    int normalizedX2 =
        (originalBox.x2 * 1000 / image.width).round().clamp(0, 1000);
    int normalizedY2 =
        (originalBox.y2 * 1000 / image.height).round().clamp(0, 1000);

    return BoundingBox(
      x1: normalizedX1,
      y1: normalizedY1,
      x2: normalizedX2,
      y2: normalizedY2,
    );
  }

  /// Public method to get image dimensions (width, height)
  static Future<ui.Image?> getImageDimensions(File imageFile) async {
    return await _getImageDimensions(imageFile);
  }

  // Get the dimensions of an image
  static Future<ui.Image?> _getImageDimensions(File imageFile) async {
    try {
      final Uint8List bytes = await imageFile.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(bytes);
      final ui.FrameInfo frame = await codec.getNextFrame();
      return frame.image;
    } catch (e) {
      // Silent error - just return null
      return null;
    }
  }

  /// Crop an image using the bounding box coordinates
  /// Returns the path to the cropped image file
  static Future<String> cropImageWithBoundingBox(
      String imagePath, BoundingBox boundingBox) async {
    try {
      // Get original image dimensions
      final File imageFile = File(imagePath);
      final ui.Image? image = await _getImageDimensions(imageFile);

      if (image == null) {
        throw Exception("Failed to load image dimensions");
      } // Ensure the bounding box coordinates are valid
      BoundingBox validBox = BoundingBox(
        x1: boundingBox.x1.clamp(0, 1000),
        y1: boundingBox.y1.clamp(0, 1000),
        x2: boundingBox.x2.clamp(0, 1000),
        y2: boundingBox.y2.clamp(0, 1000),
      );

      // Make sure x2 > x1 and y2 > y1
      if (validBox.x2 <= validBox.x1) {
        validBox = BoundingBox(
            x1: validBox.x1,
            y1: validBox.y1,
            x2: (validBox.x1 + 10).clamp(0, 1000),
            y2: validBox.y2);
      }
      if (validBox.y2 <= validBox.y1) {
        validBox = BoundingBox(
            x1: validBox.x1,
            y1: validBox.y1,
            x2: validBox.x2,
            y2: (validBox.y1 + 10).clamp(0, 1000));
      }

      // Convert normalized coordinates (0-1000) to actual pixel positions
      final int x = (validBox.x1 * image.width / 1000).round();
      final int y = (validBox.y1 * image.height / 1000).round();
      final int width =
          ((validBox.x2 - validBox.x1) * image.width / 1000).round();
      final int height = ((validBox.y2 - validBox.y1) * image.height / 1000)
          .round(); // Debug information
      debugPrint('Image dimensions: ${image.width}x${image.height}');
      debugPrint(
          'Cropping with bounds: ($x, $y) - width: $width, height: $height)');

      // Load the image
      final Uint8List bytes = await imageFile.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(bytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();

      // Create a recorder to draw the cropped portion
      final ui.PictureRecorder recorder = ui.PictureRecorder();
      final Canvas canvas = Canvas(recorder);

      // Draw only the cropped portion
      canvas.drawImageRect(
        frameInfo.image,
        Rect.fromLTWH(
            x.toDouble(), y.toDouble(), width.toDouble(), height.toDouble()),
        Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
        Paint(),
      );

      // Convert to image
      final ui.Image croppedImage =
          await recorder.endRecording().toImage(width, height);

      // Convert to bytes
      final ByteData? byteData =
          await croppedImage.toByteData(format: ui.ImageByteFormat.png);
      final Uint8List pngBytes = byteData!.buffer.asUint8List();

      // Save to file
      final directory = await getTemporaryDirectory();
      final String fileName =
          'cropped_${DateTime.now().millisecondsSinceEpoch}.png';
      final File croppedFile = File('${directory.path}/$fileName');
      await croppedFile.writeAsBytes(pngBytes);

      return croppedFile.path;
    } catch (e) {
      throw Exception('Error cropping image: $e');
    }
  }

  /// Creates a default centered bounding box for manual selection
  /// The box will be centered with 50% of the image size
  static Future<BoundingBox> createDefaultBoundingBox(String imagePath) async {
    try {
      // Get image dimensions
      final File imageFile = File(imagePath);
      final ui.Image? image = await _getImageDimensions(imageFile);

      if (image == null) {
        // Return a default box if we can't get image dimensions
        return BoundingBox(
          x1: 250, // 25% from left
          y1: 250, // 25% from top
          x2: 750, // 75% from left
          y2: 750, // 75% from top
        );
      }

      // Create a box that's centered and covers 50% of the image
      // Using normalized coordinates (0-1000 range)
      return BoundingBox(
        x1: 250, // 25% from left
        y1: 250, // 25% from top
        x2: 750, // 75% from left
        y2: 750, // 75% from top
      );
    } catch (e) {
      // Return a default box if there's an error
      return BoundingBox(
        x1: 250,
        y1: 250,
        x2: 750,
        y2: 750,
      );
    }
  }
}
