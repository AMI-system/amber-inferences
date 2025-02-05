import os
import warnings
from datetime import datetime
import json
import numpy as np
import pandas as pd
import boto3
import torch
import torchvision.transforms as transforms
from PIL import Image
from boto3.s3.transfer import TransferConfig

# ignore the pandas Future Warning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Transfer configuration for optimised S3 download
transfer_config = TransferConfig(
    max_concurrency=20,  # Increase the number of concurrent transfers
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_io_queue=1000,
    io_chunksize=262144,  # 256KB
)

def get_boxes(localisation_model, image, image_path, original_width, original_height, proc_device):
    if type(localisation_model).__name__ == 'FasterRCNN':
        # Standard localisation model
        transform_loc = transforms.Compose(
                [
                    transforms.Resize((300, 300)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        input_tensor = transform_loc(image).unsqueeze(0).to(proc_device)
        with torch.no_grad():
            localisation_outputs = localisation_model(input_tensor)
        localisation_outputs = localisation_outputs[0]

        box_coords = []
        for i in range(len(localisation_outputs["boxes"])):
            x_min, y_min, x_max, y_max = localisation_outputs["boxes"][i]

            x_min = float(x_min) * original_width / 300
            y_min = float(y_min) * original_height / 300
            x_max = float(x_max) * original_width / 300
            y_max = float(y_max) * original_height / 300

            box_coords = box_coords + [[x_min, y_min, x_max, y_max]]

    else:
        # flatbug model
        localisation_outputs = flatbug(image_path, localisation_model)
        box_coords = localisation_outputs["boxes"]

    return [localisation_outputs, box_coords]

def flatbug(image_path, flatbug_model):
    output = flatbug_model(image_path)

    # Save a visualization of the predictions
    if len(output.json_data["boxes"]) > 0:
        print(f"Saving annotated image: {image_path}")
        output.plot(
            outpath=f"{os.path.dirname(image_path)}/flatbug/flatbug_{os.path.basename(image_path)}"
        )

    # rename the confs item as scores
    crop_info = output.json_data
    crop_info["scores"] = crop_info.pop("confs")
    crop_info["labels"] = crop_info.pop("classes")

    return crop_info


def classify_species(image_tensor, regional_model, regional_category_map, top_n=5):
    """
    Classify the species of the moth using the regional model.
    """

    # print('Inference for species...')
    output = regional_model(image_tensor)
    predictions = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]

    # Sort predictions to get the indices of the top 5 scores
    top_n_indices = predictions.argsort()[-top_n:][::-1]

    # Map indices to labels and fetch their confidence scores
    index_to_label = {index: label for label, index in regional_category_map.items()}
    top_n_labels = [index_to_label[idx] for idx in top_n_indices]
    top_n_scores = [predictions[idx] for idx in top_n_indices]

    return top_n_labels, top_n_scores


def classify_order(image_tensor, order_model, order_labels, order_data_thresholds):
    """
    Classify the order of the object using the order model by Bjerge et al.
    Model and code available at: https://github.com/kimbjerge/MCC24-trap/tree/main
    """

    # print('Inference for order...')
    pred = order_model(image_tensor)
    pred = torch.nn.functional.softmax(pred, dim=1)
    predictions = pred.cpu().detach().numpy()

    predicted_label = np.argmax(predictions, axis=1)[0]
    score = predictions.max(axis=1).astype(float)[0]

    label = order_labels[predicted_label]

    return label, score


def classify_box(image_tensor, binary_model):
    """
    Classify the object as moth or non-moth using the binary model.
    """

    # print('Inference for moth/non-moth...')
    output = binary_model(image_tensor)

    predictions = torch.nn.functional.softmax(output, dim=1)
    predictions = predictions.cpu().detach().numpy()
    categories = predictions.argmax(axis=1)

    labels = {"moth": 0, "nonmoth": 1}

    index_to_label = {index: label for label, index in labels.items()}
    label = [index_to_label[cat] for cat in categories][0]
    score = predictions.max(axis=1).astype(float)[0]
    return label, score


def perform_inf(
    image_path,
    bucket_name,
    localisation_model,
    binary_model,
    order_model,
    order_labels,
    regional_model,
    regional_category_map,
    proc_device,
    order_data_thresholds,
    csv_file,
    save_crops,
    box_threshold=0.995,
    top_n=5,
):
    """
    Perform inferences on an image including:
    - object detection (localisation)
    - order classification
    - species classification
    """

    transform_species = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    all_cols = [
        "image_path",
        "image_datetime",
        "bucket_name",
        "analysis_datetime",
        "crop_status",
        "box_score",
        "box_label",
        "x_min",
        "y_min",
        "x_max",
        "y_max",  # localisation info
        "class_name",
        "class_confidence",  # binary class info
        "order_name",
        "order_confidence",  # order info
        "cropped_image_path",
    ]
    all_cols = (
        all_cols
        + ["top_" + str(i + 1) + "_species" for i in range(top_n)]
        + ["top_" + str(i + 1) + "_confidence" for i in range(top_n)]
    )

    # extract the datetime from the image path
    image_dt = os.path.basename(image_path).split("-")[0]
    image_dt = datetime.strptime(image_dt, "%Y%m%d%H%M%S%f")
    image_dt = datetime.strftime(image_dt, "%Y-%m-%d %H:%M:%S")

    current_dt = datetime.now()
    current_dt = datetime.strftime(current_dt, "%Y-%m-%d %H:%M:%S")

    try:
        # check if image_path viable
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_path = os.path.abspath(image_path)
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

        df = pd.DataFrame(
            [
                [image_path, image_dt, bucket_name, current_dt, "IMAGE CORRUPT"]
                + [""] * (len(all_cols) - 5),
            ],
            columns=all_cols,
        )

        df.to_csv(
            f"{csv_file}",
            mode="a",
            header=not os.path.isfile(csv_file),
            index=False,
        )
        return  # Skip this image

    original_image = image.copy()
    original_width, original_height = image.size

    print('Inference for the localisation model...')
    localisation_outputs, box_coords = get_boxes(localisation_model, image, image_path, original_width, original_height, proc_device)

    skipped = []

    # catch no crops: if no boxes or all boxes below threshold
    if len(box_coords) == 0 or all(
        [score < box_threshold for score in localisation_outputs["scores"]]
    ):
        skipped = [True]

    # for each detection
    for i in range(0, len(box_coords)):
        crop_status = "crop " + str(i)
        print(crop_status)
        x_min, y_min, x_max, y_max = box_coords[i]

        box_score = localisation_outputs["scores"][i]
        box_label = localisation_outputs["labels"][i]

        if box_score < box_threshold:
            continue

        # Crop the detected region and perform classification
        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
        cropped_tensor = transform_species(cropped_image).unsqueeze(0).to(proc_device)

        class_name, class_confidence = classify_box(cropped_tensor, binary_model)
        order_name, order_confidence = classify_order(
            cropped_tensor, order_model, order_labels, order_data_thresholds
        )

        # Annotate image with bounding box and class
        if class_name == "moth" or "Lepidoptera" in order_name:
            print("species classifier")
            species_names, species_confidences = classify_species(
                cropped_tensor, regional_model, regional_category_map, top_n
            )

        else:
            species_names, species_confidences = [""] * top_n, [""] * top_n

        # if save_crops then save the cropped image
        crop_path = ""
        if save_crops and i > 0:
            crop_path = image_path.replace(".jpg", f"_crop{i}.jpg")
            print(crop_path)
            cropped_image.save(crop_path)

        # append to csv with pandas
        df = pd.DataFrame(
            [
                [
                    image_path,
                    image_dt,
                    bucket_name,
                    current_dt,
                    crop_status,
                    box_score,
                    box_label,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    class_name,
                    class_confidence,
                    order_name,
                    order_confidence,
                    crop_path,
                ]
                + species_names
                + species_confidences
            ],
            columns=all_cols,
        )

        df.to_csv(
            f"{csv_file}",
            mode="a",
            header=not os.path.isfile(csv_file),
            index=False,
        )

    # catch images where no detection or all considered too large/not confident enough
    if all(skipped):
        df = pd.DataFrame(
            [
                [
                    image_path,
                    image_dt,
                    bucket_name,
                    current_dt,
                    "NO DETECTIONS FOR IMAGE",
                ]
                + [""] * (len(all_cols) - 5),
            ],
            columns=all_cols,
        )
        df.to_csv(
            f"{csv_file}",
            mode="a",
            header=not os.path.isfile(csv_file),
            index=False,
        )


def initialise_session(credentials_file="credentials.json"):
    """
    Load AWS and API credentials from a configuration file and initialise an AWS session.

    Args:
        credentials_file (str): Path to the credentials JSON file.

    Returns:
        boto3.Client: Initialised S3 client.
    """
    with open(credentials_file, encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])
    return client


def download_and_analyse(
    keys,
    output_dir,
    bucket_name,
    client,
    remove_image=True,
    perform_inference=True,
    save_crops=False,
    localisation_model=None,
    box_threshold=0.99,
    binary_model=None,
    order_model=None,
    order_labels=None,
    species_model=None,
    species_labels=None,
    device=None,
    order_data_thresholds=None,
    top_n=5,
    csv_file="results.csv",
):
    """
    Download images from S3 and perform analysis.

    Args:
        keys (list): List of S3 keys to process.
        output_dir (str): Directory to save downloaded files and results.
        bucket_name (str): S3 bucket name.
        client (boto3.Client): Initialised S3 client.
        Other args: Parameters for inference and analysis.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for key in keys:
        local_path = os.path.join(output_dir, os.path.basename(key))
        print(f"Downloading {key} to {local_path}")
        client.download_file(bucket_name, key, local_path, Config=transfer_config)

        # Perform image analysis if enabled
        print(f"Analysing {local_path}")
        if perform_inference:
            perform_inf(
                local_path,
                bucket_name=bucket_name,
                localisation_model=localisation_model,
                box_threshold=box_threshold,
                binary_model=binary_model,
                order_model=order_model,
                order_labels=order_labels,
                regional_model=species_model,
                regional_category_map=species_labels,
                proc_device=device,
                order_data_thresholds=order_data_thresholds,
                csv_file=csv_file,
                top_n=top_n,
                save_crops=save_crops,
            )
        # Remove the image if cleanup is enabled
        if remove_image:
            os.remove(local_path)

