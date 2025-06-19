from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as plt


def image_annotation(image_path, img=None, boxes={}):
    if img is None:
        image_path = Path(image_path)
        img = Image.open(image_path)

    draw = ImageDraw.Draw(img)
    for box in boxes:
        x0 = box["x_min"]
        y0 = box["y_min"]
        x1 = box["x_max"]
        y1 = box["y_max"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        if "label" in box.keys():
            draw.text((x0, y0), box["label"], fill="red")

    plt.imshow(img)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_gbif_image(species_name):
    """Fetches the first available image from GBIF for a given species."""
    search_url = f"https://api.gbif.org/v1/species/match?name={species_name}"
    response = requests.get(search_url).json()

    if "usageKey" not in response:
        print(f"Species '{species_name}' not found.")
        return None

    species_key = response["usageKey"]
    occurrence_url = f"https://api.gbif.org/v1/occurrence/search?taxonKey={species_key}&mediaType=StillImage&limit=10"
    occurrence_data = requests.get(occurrence_url).json()

    if "results" not in occurrence_data or not occurrence_data["results"]:
        print(f"No images found for '{species_name}'.")
        return None

    # Extract the first image available
    for record in occurrence_data["results"]:
        if "media" in record and record["media"]:
            image_url = record["media"][0]["identifier"]
            try:
                img_data = requests.get(image_url).content
                img = Image.open(BytesIO(img_data))
                return img
            except Exception as e:
                print(f"Error loading image for {species_name}: {e}")

    print(f"No valid images found for '{species_name}'.")
    return None
