import os
from PIL import ImageDraw, Image, ImageFont
from pathlib import Path


def _get_font():
    try:
        if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        else:
            import cv2

            font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        return ImageFont.truetype(font_path, size=50)
    except Exception as e:
        print(f"Loading default font, could not another: {e}")
        return ImageFont.load_default()


def _normalize_box(box, img, scale, default_colour):
    x0 = float(box["x_min"])
    y0 = float(box["y_min"])
    x1 = float(box["x_max"])
    y1 = float(box["y_max"])
    if scale:
        og_width, og_height = img.size
        x0 = x0 / 300 * og_width
        y0 = y0 / 300 * og_height
        x1 = x1 / 300 * og_width
        y1 = y1 / 300 * og_height
    ann_col = box.get("ann_col", default_colour)
    label = box.get("label", "")
    return (x0, y0, x1, y1, ann_col, label)


def image_annotation(
    image_path, img=None, boxes={}, scale=False, default_colour="grey"
):
    """
    Annotates an image with bounding boxes and labels.
    Args:
        image_path (str or Path): Path to the image file.
        img (PIL.Image.Image, optional): Image object. If None, the image will be loaded from `image_path`.
        boxes (list of dict): List of bounding boxes, each represented as a dictionary with keys:
            - "x_min", "y_min", "x_max", "y_max": Coordinates of the bounding box.
            - "label": Text label for the bounding box (optional).
            - "ann_col": Color for the bounding box outline (optional, defaults to "grey").
        scale (bool): If True, scales the bounding box coordinates based on the original image size.
        default_colour (str): Default color for bounding boxes if not specified in the box dict.
    Returns:
        PIL.Image.Image: Annotated image.
    """
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list of dictionaries")
    if not all(isinstance(box, dict) for box in boxes):
        raise ValueError("Each box must be a dictionary")

    if img is None:
        image_path = Path(image_path)
        img = Image.open(image_path)

    draw = ImageDraw.Draw(img)
    font = _get_font()

    for box in boxes:
        x0, y0, x1, y1, ann_col, label = _normalize_box(box, img, scale, default_colour)
        draw.rectangle([x0, y0, x1, y1], outline=ann_col, width=3)
        draw.text((x0, y0), label, fill=ann_col, font=font)

    return img


def gif_creater(input_dir, output_path):
    """
    Creates a GIF from a sequence of images in a directory.
    Args:
        input_dir (str or Path): Directory containing images to be converted to GIF.
        output_path (str or Path): Path where the GIF will be saved.
    """
    if not isinstance(input_dir, (str, Path)):
        raise ValueError("input_dir must be a string or Path object")
    if not isinstance(output_path, (str, Path)):
        raise ValueError("output_path must be a string or Path object")

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory")
    if not output_path.suffix.lower() == ".gif":
        raise ValueError(f"{output_path} must have a .gif extension")
    if not output_path.parent.exists():
        raise ValueError(f"Output directory {output_path.parent} does not exist")
    if not output_path.parent.is_dir():
        raise ValueError(f"Output path {output_path.parent} is not a directory")

    # Open images and convert to a sequence
    image_paths = sorted([p for p in input_dir.iterdir() if p.is_file()])
    images = [Image.open(img) for img in image_paths]

    # Save as GIF
    images[0].save(
        output_path, save_all=True, append_images=images[1:], duration=500, loop=0
    )

    del images
