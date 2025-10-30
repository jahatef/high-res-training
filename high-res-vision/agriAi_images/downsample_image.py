from PIL import Image
import argparse
import os

def downsample_to_384(image_path, output_path=None):
    """
    Downsample an image to 384x384 pixels and save as PNG.
    """
    img = Image.open(image_path)
    resized = img.resize((384, 384), Image.LANCZOS)  # High-quality downsampling

    # Default output path
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + "_384.png"

    resized.save(output_path, format="PNG")
    print(f"✅ Saved downsampled image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample an image to 384x384 pixels and save as PNG.")
    parser.add_argument("--image_path", required=True, help="Path to the input image.")
    parser.add_argument("--output_path", default=None, help="Optional path to save the resized PNG.")
    args = parser.parse_args()

    downsample_to_384(args.image_path, args.output_path)

