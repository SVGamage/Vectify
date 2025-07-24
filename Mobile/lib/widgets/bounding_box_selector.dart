import 'dart:io';
import 'package:flutter/material.dart';
import '../models/detection_model.dart';
import '../utils/image_utils.dart';

class BoundingBoxSelector extends StatefulWidget {
  final String imagePath;
  final List<Detection> detections;
  final Function(BoundingBox) onBoundingBoxChanged;
  final bool initialEditingMode;

  const BoundingBoxSelector({
    super.key,
    required this.imagePath,
    required this.detections,
    required this.onBoundingBoxChanged,
    this.initialEditingMode = false,
  });

  @override
  _BoundingBoxSelectorState createState() => _BoundingBoxSelectorState();
}

class _BoundingBoxSelectorState extends State<BoundingBoxSelector> {
  BoundingBox? _currentBoundingBox;
  bool _isEditing = false;

  // Keep track of resize handles being dragged
  bool _isDraggingTopLeft = false;
  bool _isDraggingTopRight = false;
  bool _isDraggingBottomLeft = false;
  bool _isDraggingBottomRight = false;
  bool _isDraggingBox = false;

  // Offset for box dragging
  Offset _dragStartOffset = Offset.zero;
  late BoundingBox _originalBox;

  // Widget display size
  double _imageWidth = 0;
  double _imageHeight = 0;
  // Real image pixel size
  int _realImageWidth = 0;
  int _realImageHeight = 0;

  @override
  void initState() {
    super.initState();
    // Set initial editing mode
    _isEditing = widget.initialEditingMode;

    // Initialize with the first detection if available
    if (widget.detections.isNotEmpty) {
      _initializeBoundingBox();
    }
    _fetchRealImageDimensions();
  }

  Future<void> _fetchRealImageDimensions() async {
    final file = File(widget.imagePath);
    final img = await ImageUtils.getImageDimensions(file);
    if (img != null && mounted) {
      setState(() {
        _realImageWidth = img.width;
        _realImageHeight = img.height;
      });
    }
  }

  Future<void> _initializeBoundingBox() async {
    // Get the original bounding box
    final originalBox = widget.detections.first.boundingBox;

    // Check if the coordinates might be in pixels rather than normalized
    if (originalBox.x2 > 1000 || originalBox.y2 > 1000) {
      // Convert from pixel coordinates to 0-1000 range
      final normalizedBox = await ImageUtils.convertToNormalizedBoundingBox(
          widget.imagePath, originalBox);
      if (mounted) {
        setState(() {
          _currentBoundingBox = normalizedBox;
        });
      }
    } else {
      if (mounted) {
        setState(() {
          _currentBoundingBox = originalBox;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return GestureDetector(
          onPanDown: _handlePanDown,
          onPanUpdate: _handlePanUpdate,
          onPanEnd: _handlePanEnd,
          child: Stack(
            children: [
              // Image
              Image.file(
                File(widget.imagePath),
                fit: BoxFit.contain,
                width: double.infinity,
                height: double.infinity,
                frameBuilder: (context, child, frame, wasSynchronouslyLoaded) {
                  // Only update dimensions once when image is fully loaded
                  if (frame != null && _imageWidth == 0 && _imageHeight == 0) {
                    _updateImageDimensions(context);
                  }
                  return child;
                },
              ),

              // Bounding box overlay
              if (_currentBoundingBox != null) _buildBoundingBoxOverlay(),

              // Edit button
              Positioned(
                top: 10,
                right: 10,
                child: ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _isEditing = !_isEditing;
                    });
                  },
                  child: Text(_isEditing ? 'Done' : 'Edit Box'),
                ),
              ),

              // Instructions for manual editing (when in edit mode)
              if (_isEditing)
                Positioned(
                  bottom: 20,
                  left: 20,
                  right: 20,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          'Adjust Logo Selection',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        SizedBox(height: 8),
                        Text(
                          '• Drag the corners to resize\n• Drag the box to reposition\n• Tap "Done" when finished',
                          style: TextStyle(color: Colors.white),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  void _updateImageDimensions(BuildContext context) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;

      final RenderBox? renderBox = context.findRenderObject() as RenderBox?;
      if (renderBox == null) return;

      final size = renderBox.size;

      // Only update dimensions if they've changed
      if (size.width != _imageWidth || size.height != _imageHeight) {
        setState(() {
          _imageWidth = size.width;
          _imageHeight = size.height;
        });
      }
    });
  }

  // Helper to get the image's drawn area (offset and size) inside the widget
  Rect _getImageDrawnRect() {
    if (_imageWidth == 0 ||
        _imageHeight == 0 ||
        _realImageWidth == 0 ||
        _realImageHeight == 0) {
      return Rect.zero;
    }
    final widgetAspect = _imageWidth / _imageHeight;
    final imageAspect = _realImageWidth / _realImageHeight;
    double drawWidth, drawHeight, dx, dy;
    if (imageAspect > widgetAspect) {
      // Image is wider than widget: fit width
      drawWidth = _imageWidth;
      drawHeight = _imageWidth / imageAspect;
      dx = 0;
      dy = (_imageHeight - drawHeight) / 2;
    } else {
      // Image is taller than widget: fit height
      drawHeight = _imageHeight;
      drawWidth = _imageHeight * imageAspect;
      dx = (_imageWidth - drawWidth) / 2;
      dy = 0;
    }
    return Rect.fromLTWH(dx, dy, drawWidth, drawHeight);
  }

  // Map widget/touch coordinates to normalized image coordinates (0-1000)
  Offset _widgetToNormalized(Offset localOffset) {
    final rect = _getImageDrawnRect();
    final dx = ((localOffset.dx - rect.left) / rect.width).clamp(0.0, 1.0);
    final dy = ((localOffset.dy - rect.top) / rect.height).clamp(0.0, 1.0);
    return Offset(dx * 1000, dy * 1000);
  }

  // Map normalized image coordinates (0-1000) to widget coordinates
  Offset _normalizedToWidget(double x, double y) {
    final rect = _getImageDrawnRect();
    return Offset(
      rect.left + (x / 1000) * rect.width,
      rect.top + (y / 1000) * rect.height,
    );
  }

  // --- Update pan handlers to use new mapping ---
  void _handlePanDown(DragDownDetails details) {
    if (!_isEditing || _currentBoundingBox == null) return;
    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset localOffset = box.globalToLocal(details.globalPosition);
    final currentBox = _currentBoundingBox!;
    _originalBox = currentBox;
    _dragStartOffset = localOffset;
    // Convert touch position to normalized coordinates (0-1000) using drawn area
    final Offset norm = _widgetToNormalized(localOffset);
    final double touchX = norm.dx;
    final double touchY = norm.dy;

    // Use API coordinates directly
    final double left = currentBox.x1.toDouble();
    final double top = currentBox.y1.toDouble();
    final double width = (currentBox.x2 - currentBox.x1).toDouble();
    final double height = (currentBox.y2 - currentBox.y1).toDouble();

    // Check if user is dragging a resize handle
    // Define handle range in normalized coordinates
    const double handleRange = 30.0;
    // Convert handle range to API coordinates
    final double scaledHandleRange = handleRange; // already in normalized units

    if ((touchX - left).abs() < scaledHandleRange &&
        (touchY - top).abs() < scaledHandleRange) {
      _isDraggingTopLeft = true;
    } else if ((touchX - (left + width)).abs() < scaledHandleRange &&
        (touchY - top).abs() < scaledHandleRange) {
      _isDraggingTopRight = true;
    } else if ((touchX - left).abs() < scaledHandleRange &&
        (touchY - (top + height)).abs() < scaledHandleRange) {
      _isDraggingBottomLeft = true;
    } else if ((touchX - (left + width)).abs() < scaledHandleRange &&
        (touchY - (top + height)).abs() < scaledHandleRange) {
      _isDraggingBottomRight = true;
    } else if (touchX >= left &&
        touchX <= left + width &&
        touchY >= top &&
        touchY <= top + height) {
      _isDraggingBox = true;
    }
  }

  void _handlePanUpdate(DragUpdateDetails details) {
    if (!_isEditing || _currentBoundingBox == null) return;
    if (!(_isDraggingTopLeft ||
        _isDraggingTopRight ||
        _isDraggingBottomLeft ||
        _isDraggingBottomRight ||
        _isDraggingBox)) {
      return;
    }
    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset localOffset = box.globalToLocal(details.globalPosition);
    final Offset startNorm = _widgetToNormalized(_dragStartOffset);
    final Offset currNorm = _widgetToNormalized(localOffset);
    final double scaledDx = currNorm.dx - startNorm.dx;
    final double scaledDy = currNorm.dy - startNorm.dy;
    setState(() {
      if (_isDraggingBox) {
        _currentBoundingBox = BoundingBox(
          x1: (_originalBox.x1 + scaledDx).round().clamp(0, 1000),
          y1: (_originalBox.y1 + scaledDy).round().clamp(0, 1000),
          x2: (_originalBox.x2 + scaledDx).round().clamp(0, 1000),
          y2: (_originalBox.y2 + scaledDy).round().clamp(0, 1000),
        );
      } else {
        int newX1 = _originalBox.x1;
        int newY1 = _originalBox.y1;
        int newX2 = _originalBox.x2;
        int newY2 = _originalBox.y2;
        if (_isDraggingTopLeft) {
          newX1 = (_originalBox.x1 + scaledDx)
              .round()
              .clamp(0, _originalBox.x2 - 10);
          newY1 = (_originalBox.y1 + scaledDy)
              .round()
              .clamp(0, _originalBox.y2 - 10);
        } else if (_isDraggingTopRight) {
          newX2 = (_originalBox.x2 + scaledDx)
              .round()
              .clamp(_originalBox.x1 + 10, 1000);
          newY1 = (_originalBox.y1 + scaledDy)
              .round()
              .clamp(0, _originalBox.y2 - 10);
        } else if (_isDraggingBottomLeft) {
          newX1 = (_originalBox.x1 + scaledDx)
              .round()
              .clamp(0, _originalBox.x2 - 10);
          newY2 = (_originalBox.y2 + scaledDy)
              .round()
              .clamp(_originalBox.y1 + 10, 1000);
        } else if (_isDraggingBottomRight) {
          newX2 = (_originalBox.x2 + scaledDx)
              .round()
              .clamp(_originalBox.x1 + 10, 1000);
          newY2 = (_originalBox.y2 + scaledDy)
              .round()
              .clamp(_originalBox.y1 + 10, 1000);
        }
        _currentBoundingBox = BoundingBox(
          x1: newX1,
          y1: newY1,
          x2: newX2,
          y2: newY2,
        );
      }

      // Notify parent of the change
      widget.onBoundingBoxChanged(_currentBoundingBox!);
    });
  }

  void _handlePanEnd(DragEndDetails details) {
    _isDraggingTopLeft = false;
    _isDraggingTopRight = false;
    _isDraggingBottomLeft = false;
    _isDraggingBottomRight = false;
    _isDraggingBox = false;
  }

  Widget _buildBoundingBoxOverlay() {
    if (_currentBoundingBox == null) {
      return Container();
    }

    final box = _currentBoundingBox!;
    final Offset topLeft =
        _normalizedToWidget(box.x1.toDouble(), box.y1.toDouble());
    final Offset bottomRight =
        _normalizedToWidget(box.x2.toDouble(), box.y2.toDouble());
    final double left = topLeft.dx;
    final double top = topLeft.dy;
    final double width = bottomRight.dx - topLeft.dx;
    final double height = bottomRight.dy - topLeft.dy;

    return Stack(
      children: [
        Positioned.fill(
          child: CustomPaint(
            painter: _OverlayPainter(
              left: left,
              top: top,
              width: width,
              height: height,
            ),
          ),
        ),
        Positioned(
          left: left,
          top: top,
          width: width,
          height: height,
          child: Container(
            decoration: BoxDecoration(
              border: Border.all(
                color: Colors.red,
                width: 3,
              ),
            ),
            child: _isEditing
                ? Stack(
                    children: [
                      _buildResizeHandle(Alignment.topLeft),
                      _buildResizeHandle(Alignment.topRight),
                      _buildResizeHandle(Alignment.bottomLeft),
                      _buildResizeHandle(Alignment.bottomRight),
                    ],
                  )
                : null,
          ),
        ),
      ],
    );
  }

  Widget _buildResizeHandle(Alignment alignment) {
    return Align(
      alignment: alignment,
      child: Container(
        width: 24, // Handle size in screen pixels - slightly larger
        height: 24,
        decoration: BoxDecoration(
          color: Colors.red.withOpacity(0.7),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 2),
        ),
      ),
    );
  }
}

// Add a custom painter for the overlay
class _OverlayPainter extends CustomPainter {
  final double left, top, width, height;
  _OverlayPainter(
      {required this.left,
      required this.top,
      required this.width,
      required this.height});
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black.withOpacity(0.5)
      ..style = PaintingStyle.fill;
    // Draw overlay with a transparent rectangle for the bounding box
    final rect = Rect.fromLTWH(left, top, width, height);
    final path = Path.combine(
      PathOperation.difference,
      Path()..addRect(Rect.fromLTWH(0, 0, size.width, size.height)),
      Path()..addRect(rect),
    );
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
