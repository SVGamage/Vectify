import 'dart:io';
import 'package:http/http.dart' as http;

class VectorizeService {
  static const String apiUrl =
      'https://vectifyfastapi-production.up.railway.app/vectorize';

  /// Sends a cropped image file to the API for vectorization
  /// Returns the SVG string if successful
  static Future<String> vectorizeImage(File imageFile) async {
    try {
      // Send the image as raw bytes (octet-stream)
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
      final svgString = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return svgString;
      } else {
        throw Exception(
            'Failed to vectorize image: \\${response.statusCode} - \\${svgString}');
      }
    } catch (e) {
      throw Exception('Error during vectorization: $e');
    }
  }
}
