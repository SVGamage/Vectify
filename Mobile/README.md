# Vectify Flutter App

A Flutter application for detecting and cropping logos from images.

## Features

- **Automatic Logo Detection**: Uses AI to detect logos in images
- **Manual Logo Selection**: Allows manual selection when automatic detection fails
- **Logo Cropping**: Crops selected logos with precise bounding box control
- **Preview & Edit**: View and edit the cropped result before saving

## How to Use

### Automatic Detection

1. Launch the app
2. Select an image from gallery or take a photo with camera
3. The app will automatically detect logos in the image
4. Adjust the bounding box if needed
5. Tap "Crop" to crop the logo
6. Save the cropped logo

### Manual Selection

If no logo is detected automatically, you can manually select a logo area:

1. When prompted that no logo was detected, tap "Select Manually"
2. A default selection box will appear centered on the image
3. Adjust the red box to select your logo:
   - Drag the corners to resize
   - Drag inside the box to move it
4. Tap "Crop" in the top bar to crop the selected area
5. Preview the result and save

## Technical Details

- Coordinates are normalized in 0-1000 range for API compatibility
- The app properly converts between screen coordinates and API coordinates
- Both automatic and manual selection use the same cropping pipeline
