from PIL import Image
import argparse
import os

def center_crop_384(image_path, output_path=None):
    """
    Center-crop an image to 384x384 pixels and save as PNG.
    """
    img = Image.open(image_path)
    width, height = img.size
    crop_size = 4096

    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    cropped = img.crop((left, top, right, bottom))

    # Default output path
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + "_center384.png"

    cropped.save(output_path, format="PNG")
    print(f"✅ Saved cropped image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center crop an image to 384x384 pixels and save as PNG.")
    parser.add_argument("--image_path", required=True, help="Path to the input image.")
    parser.add_argument("--output_path", default=None, help="Optional path to save the cropped PNG.")
    args = parser.parse_args()

    center_crop_384(args.image_path, args.output_path)

