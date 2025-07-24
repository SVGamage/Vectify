// Models for the API response
class DetectionResponse {
  final bool success;
  final List<Detection> detections;
  final String annotatedImagePath;

  DetectionResponse({
    required this.success,
    required this.detections,
    required this.annotatedImagePath,
  });

  factory DetectionResponse.fromJson(Map<String, dynamic> json) {
    List<dynamic> detectionsJson = json['detections'] ?? [];
    List<Detection> detectionsList = detectionsJson
        .map((detection) => Detection.fromJson(detection))
        .toList();

    return DetectionResponse(
      success: json['success'] ?? false,
      detections: detectionsList,
      annotatedImagePath: json['annotated_image_path'] ?? '',
    );
  }
}

class Detection {
  final BoundingBox boundingBox;
  final int classId;
  final String className;
  final double confidence;

  Detection({
    required this.boundingBox,
    required this.classId,
    required this.className,
    required this.confidence,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      boundingBox: BoundingBox.fromJson(json['bounding_box']),
      classId: json['class_id'] ?? 0,
      className: json['class_name'] ?? '',
      confidence: (json['confidence'] ?? 0.0).toDouble(),
    );
  }
}

class BoundingBox {
  final int x1;
  final int y1;
  final int x2;
  final int y2;

  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: json['x1'] ?? 0,
      y1: json['y1'] ?? 0,
      x2: json['x2'] ?? 0,
      y2: json['y2'] ?? 0,
    );
  }
}
