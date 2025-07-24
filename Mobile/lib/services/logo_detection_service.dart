import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/detection_model.dart';

class LogoDetectionService {
  static const String apiUrl =
      'https://vectifyfastapi-production.up.railway.app/detect';

  /// Sends an image file to the API for logo detection
  /// Returns a DetectionResponse object
  static Future<DetectionResponse> detectLogo(File imageFile) async {
    try {
      // Create a multipart request
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));

      // Add the image file to the request with key 'image'
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );

      // Send the request
      var response = await request.send();

      // Get the response as string
      var responseString = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        // Parse the response
        Map<String, dynamic> jsonResponse = jsonDecode(responseString);
        return DetectionResponse.fromJson(jsonResponse);
      } else {
        throw Exception(
            'Failed to detect logo: ${response.statusCode} - $responseString');
      }
    } catch (e) {
      throw Exception('Error during logo detection: $e');
    }
  }
}
