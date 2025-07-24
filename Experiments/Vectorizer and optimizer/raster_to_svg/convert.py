import vtracer
import os

def convert_raster_to_svg_vtracer(input_path, output_path):
    """
    Convert a raster image to SVG using VTracer with color preservation and sharp edges.
    :param input_path: Path to the input raster image (e.g., JPG).
    :param output_path: Path to save the output SVG file.
    """
    try:
        # Ensure the input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")

        # Convert using VTracer
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            colormode="color",          # Full-color mode
            hierarchical="stacked",     # Stacked shapes for compact output
            mode="spline",              # Smooth curves for sharp edges
            filter_speckle=4,           # Remove small noise (adjustable)
            color_precision=6,          # Color accuracy (6-8 bits)
            layer_difference=16,        # Color layer separation
            corner_threshold=60,        # Angle to detect corners
            length_threshold=4.0,       # Min segment length
            max_iterations=10,          # Curve fitting iterations
            splice_threshold=45,        # Spline splicing angle
            path_precision=3            # Decimal precision in paths
        )

        if os.path.exists(output_path):
            print(f"Successfully converted {input_path} to {output_path}")
        else:
            print(f"Conversion failed: {output_path} not created.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(script_dir, "3.jpg")
    output_svg = os.path.join(script_dir, "output_1.svg")

    # Run the conversion
    convert_raster_to_svg_vtracer(input_image, output_svg)