import os
import warnings
from datetime import datetime, time
import json
import numpy as np
import pandas as pd
import boto3
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pathlib import Path
from boto3.s3.transfer import TransferConfig
from amber_inferences.utils.tracking import (
    l2_normalize,
    calculate_cost,
    find_best_matches,
)

# ignore the pandas Future Warning
warnings.simplefilter(action="ignore", category=FutureWarning)

# Transfer configuration for optimised S3 download
transfer_config = TransferConfig(
    max_concurrency=20,  # Increase the number of concurrent transfers
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_io_queue=1000,
    io_chunksize=262144,  # 256KB
)


def get_boxes(
    localisation_model, image, image_path, original_width, original_height, proc_device
):
    if type(localisation_model).__name__ == "FasterRCNN":
        # Standard localisation model
        transform_loc = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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


def flatbug(image_path, flatbug_model, save_annotation=False):
    output = flatbug_model(str(image_path))

    # Save a visualisation of the predictions
    if len(output.json_data["boxes"]) > 0 and save_annotation:
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

    # Flatten and normalize features
    features = output.view(output.size(0), -1)  # shape: (1, N)
    features = features.squeeze(0).cpu()  # remove batch dim, move to CPU
    features = l2_normalize(features)  # normalize
    features = features.detach().numpy()  # convert to numpy

    return top_n_labels, top_n_scores, features


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


def variance_of_laplacian(image):
    # compute the Laplacian of the image
    # (second-order derivative, which highlights areas of rapid intensity change).
    # higher variance means more edges/sharpness, while low variance indicates blurriness.
    return cv2.Laplacian(image, cv2.CV_64F).var()


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


def _extract_deployment_fields(dep_data):
    """Return deployment fields in the order needed for result rows."""
    return [
        dep_data.get("country_code", ""),
        dep_data.get("location_name", ""),
        dep_data.get("deployment_id", ""),
        dep_data.get("lat", ""),
        dep_data.get("lon", ""),
    ]


def _build_crop_row(
    image_path,
    image_dt,
    dep_data,
    current_dt,
    recording_session,
    job_name,
    image_bluriness,
    crop_status,
    box_score,
    box_label,
    x_min,
    y_min,
    x_max,
    y_max,
    crop_bluriness,
    crop_area,
    crop_path,
    all_cols,
):
    """Build a result row for a detected crop."""
    return [
        image_path,
        image_dt,
        *_extract_deployment_fields(dep_data),
        current_dt,
        recording_session,
        job_name,
        image_bluriness,
        crop_status,
        box_score,
        box_label,
        x_min,
        y_min,
        x_max,
        y_max,
        crop_bluriness,
        crop_area,
        crop_path,
    ] + [""] * (len(all_cols) - 21)


def _build_error_row(
    image_path, image_dt, dep_data, current_dt, recording_session, message, all_cols
):
    """Build a result row for an error or skipped image."""
    return [
        image_path,
        image_dt,
        *_extract_deployment_fields(dep_data),
        current_dt,
        recording_session,
        "",
        message,
    ] + [""] * (len(all_cols) - 11)


def _save_crop_image(cropped_image, crop_dir, image_path, crop_status):
    """Save a cropped image and return its path."""
    crop_dir = Path(crop_dir)
    crop_path = crop_dir / image_path.with_suffix("").name
    crop_path = crop_path.with_name(f"{crop_path.name}_{crop_status}.jpg")
    cropped_image.save(crop_path)
    return crop_path


def crop_image_only(
    image_path,
    dep_data,
    localisation_model,
    proc_device,
    csv_file,
    save_crops,
    box_threshold=0.995,
    job_name=None,
    crop_dir=None,
):
    image_path = Path(image_path)
    csv_file = Path(csv_file)
    all_cols = [
        "image_path",
        "image_datetime",
        "bucket_name",
        "deployment_name",
        "deployment_id",
        "latitude",
        "longitude",
        "analysis_datetime",
        "recording_session",
        "job_name",
        "image_bluriness",
        "crop_status",
        "box_score",
        "box_label",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "crop_bluriness",
        "crop_area",
        "cropped_image_path",
    ]
    image_dt, recording_session = get_image_metadata(image_path)
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crops_df = pd.DataFrame(columns=all_cols)
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        row = _build_error_row(
            image_path,
            image_dt,
            dep_data,
            current_dt,
            recording_session,
            "Image corrupt",
            all_cols,
        )
        df = pd.DataFrame([row], columns=all_cols)
        df.to_csv(str(csv_file), mode="a", header=not csv_file.is_file(), index=False)
        crops_df = pd.concat([crops_df, df])
        return
    image_bluriness = variance_of_laplacian(np.array(image))

    original_image = image.copy()
    original_width, original_height = image.size
    localisation_outputs, box_coords = get_boxes(
        localisation_model,
        image,
        image_path,
        original_width,
        original_height,
        proc_device,
    )

    skipped = []
    if len(box_coords) == 0 or all(
        [score < box_threshold for score in localisation_outputs["scores"]]
    ):
        skipped = [True]
    for i, (x_min, y_min, x_max, y_max) in enumerate(box_coords):
        crop_status = f"crop_{i+1}"
        box_score = localisation_outputs["scores"][i]
        box_label = localisation_outputs["labels"][i]
        crop_area = (x_max - x_min) * (y_max - y_min)
        if float(box_score) >= box_threshold:
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            crop_bluriness = variance_of_laplacian(np.array(cropped_image))
            crop_path = ""
            if save_crops and crop_dir:
                crop_path = _save_crop_image(
                    cropped_image, crop_dir, image_path, crop_status
                )

            # print(crop_bluriness)
            row = _build_crop_row(
                image_path,
                image_dt,
                dep_data,
                current_dt,
                recording_session,
                job_name,
                image_bluriness,
                crop_status,
                box_score,
                box_label,
                x_min,
                y_min,
                x_max,
                y_max,
                crop_bluriness,
                crop_area,
                crop_path,
                all_cols,
            )
            df = pd.DataFrame([row], columns=all_cols)
            df.to_csv(
                str(csv_file), mode="a", header=not csv_file.is_file(), index=False
            )
            crops_df = pd.concat([crops_df, df])
            skipped.append(False)
        else:
            skipped.append(True)
    if all(skipped):
        row = _build_error_row(
            image_path,
            image_dt,
            dep_data,
            current_dt,
            recording_session,
            "No detections for image.",
            all_cols,
        )
        df = pd.DataFrame([row], columns=all_cols)
        df.to_csv(str(csv_file), mode="a", header=not csv_file.is_file(), index=False)
        crops_df = pd.concat([crops_df, df])
    return crops_df


def localisation_only(
    keys,
    output_dir,
    dep_data,
    client,
    remove_image=True,
    perform_inference=True,
    save_crops=False,
    localisation_model=None,
    box_threshold=0.99,
    device=None,
    csv_file=Path("results.csv"),
    job_name=None,
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
    output_dir = Path(output_dir)
    csv_file = Path(csv_file)
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(parents=True, exist_ok=True)
    for key in keys:
        local_path = output_dir / "snapshots" / Path(key).name
        print(f"Downloading {key} to {local_path}")
        client.download_file(
            dep_data["country_code"], key, str(local_path), Config=transfer_config
        )

        # Perform image analysis if enabled
        print(f"Analysing {local_path}")
        if perform_inference:
            crop_image_only(
                local_path,
                dep_data=dep_data,
                localisation_model=localisation_model,
                box_threshold=box_threshold,
                proc_device=device,
                csv_file=csv_file,
                save_crops=save_crops,
                job_name=job_name,
                crop_dir=output_dir / "crops" if save_crops else None,
            )
        # Remove the image if cleanup is enabled
        if remove_image:
            local_path.unlink()


def initialise_session(credentials_file=Path("credentials.json")):
    credentials_file = Path(credentials_file)
    with open(credentials_file, encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])
    return client


def download_image_from_key(s3_client, key, bucket, output_dir):
    output_dir = Path(output_dir)
    local_path = output_dir / Path(key).name
    output_dir.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(local_path))


def get_image_metadata(path, verbose=False):
    path = Path(path)
    try:
        dt_string = [x for x in path.name.split("-") if x.startswith(("202", "201"))][0]
        image_dt = datetime.strptime(dt_string, "%Y%m%d%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # if the time is after 12:00 then recording_session is current_dt, else it is the previous day
        recording_session = image_dt.split(" ")[0]
        if datetime.strptime(image_dt, "%Y-%m-%d %H:%M:%S").time() < time(12, 0, 0):
            recording_session = (
                datetime.strptime(image_dt, "%Y-%m-%d %H:%M:%S") - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")
        return image_dt, recording_session
    except Exception:
        print(
            f" - Could not extract datetime from {path}, returning date and session = ''"
        )
        return "", ""


def load_image(path):
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        return None


def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_embedding(data, image_path, verbose=False):
    image_path = Path(image_path)
    json_output_file = image_path.with_suffix(".json")
    if verbose:
        print(f" - Saving embedding to {json_output_file}")

    with open(json_output_file, "w") as f:
        json.dump(convert_ndarrays(data), f)


def save_result_row(data, columns, csv_file):
    csv_file = Path(csv_file)
    df = pd.DataFrame([data], columns=columns)
    write_type = "a" if csv_file.is_file() else "w"
    df.to_csv(csv_file, mode=write_type, header=not csv_file.is_file(), index=False)


def get_default_row(
    image_path,
    image_dt,
    dep_data,
    current_dt,
    recording_session,
    bluriness,
    message,
    all_cols,
):
    return [
        image_path,
        image_dt,
        dep_data["country_code"],  # bucket name
        dep_data["location_name"],
        dep_data["deployment_id"],
        dep_data["lat"],
        dep_data["lon"],
        current_dt,
        recording_session,
        bluriness,
        message,
    ] + [""] * (len(all_cols) - 11)


def get_previous_embedding(previous_image, verbose=False):
    try:
        if previous_image is not None:
            json_path = Path(previous_image).with_suffix(".json")
            if json_path.is_file():
                with open(str(json_path), "r") as f:
                    try:
                        previous_image_embedding = json.load(f)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f" - Error decoding JSON for {json_path}: {e}")
                        previous_image_embedding = {}
            else:
                print(f" - No previous image embedding found for {previous_image}.")
                previous_image_embedding = {}
        else:
            print(f" - No previous image embedding found for {previous_image}.")
            previous_image_embedding = {}
    except Exception as e:
        print(
            f" - Unexpected error loading previous embedding for {previous_image}: {e}"
        )
        previous_image_embedding = {}

    if verbose:
        print(
            f" - Found {len(previous_image_embedding)} crops from the previous image ({previous_image})."
        )

    return previous_image_embedding


def _get_species_and_embedding(
    class_name, order_name, cropped_tensor, regional_model, regional_category_map, top_n
):
    if class_name == "moth" or "Lepidoptera" in order_name:
        return classify_species(
            cropped_tensor, regional_model, regional_category_map, top_n
        )
    else:
        return [""] * top_n, [""] * top_n, None


def _get_best_matches(previous_image_embedding, crop_status, embedding_list):
    if len(previous_image_embedding) > 0:
        crop_similarities = pd.DataFrame({})
        for crop_1 in list(previous_image_embedding.keys()):
            c_1 = previous_image_embedding[crop_1]
            c_2 = embedding_list[crop_status]
            results_df = calculate_cost(c_1, c_2)
            crop_similarities = pd.concat([crop_similarities, results_df])
        return find_best_matches(crop_similarities)
    else:
        return pd.DataFrame(
            {
                "previous_image": [None],
                "best_match_crop": [
                    "No crops from previous image. Tracking not possible."
                ],
                "cnn_cost": [""],
                "iou_cost": [""],
                "box_ratio_cost": [""],
                "dist_ratio_cost": [""],
                "total_cost": [""],
            },
            columns=[
                "previous_image",
                "best_match_crop",
                "cnn_cost",
                "iou_cost",
                "box_ratio_cost",
                "dist_ratio_cost",
                "total_cost",
            ],
        )


def perform_inf(
    image_path,
    dep_data,
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
    verbose=False,
    previous_image=None,
    crop_dir=None,
):
    image_path = Path(image_path)
    csv_file = Path(csv_file)

    transform_species = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # load in embedding from the previous image
    previous_image_embedding = get_previous_embedding(previous_image, verbose)

    all_cols = (
        [
            "image_path",
            "image_datetime",
            "bucket_name",
            "deployment_name",
            "deployment_id",
            "latitude",
            "longitude",
            "analysis_datetime",
            "recording_session",
            "image_bluriness",
            "crop_status",
            "crop_bluriness",
            "crop_area",
            "cropped_image_path",
            "box_score",
            "box_label",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "class_name",
            "class_confidence",
            "order_name",
            "order_confidence",
        ]
        + [f"top_{i+1}_species" for i in range(top_n)]
        + [f"top_{i+1}_confidence" for i in range(top_n)]
        + [
            "previous_image",
            "best_match_crop",
            "cnn_cost",
            "iou_cost",
            "box_ratio_cost",
            "dist_ratio_cost",
            "total_cost",
        ]
    )

    image_dt, recording_session = get_image_metadata(image_path)
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    image = load_image(image_path)
    if image is None:
        save_result_row(
            get_default_row(
                image_path,
                image_dt,
                dep_data,
                current_dt,
                recording_session,
                "",
                "Image corrupt",
                all_cols,
            ),
            all_cols,
            csv_file,
        )
        save_embedding({}, image_path, verbose=verbose)
        return

    image_bluriness = variance_of_laplacian(np.array(image))
    original_image = image.copy()
    original_width, original_height = image.size

    localisation_outputs, box_coords = get_boxes(
        localisation_model,
        image,
        image_path,
        original_width,
        original_height,
        proc_device,
    )

    if verbose:
        print(f" - Found {len(box_coords)} box(es) in image {image_path}")

    skipped = True
    embedding_list = {}
    for i, (x_min, y_min, x_max, y_max) in enumerate(box_coords):
        box_score = localisation_outputs["scores"][i]
        if box_score < box_threshold:
            continue

        skipped = False
        crop_status = f"crop_{i+1}"
        crop_area = (x_max - x_min) * (y_max - y_min)
        box_label = localisation_outputs["labels"][i]

        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
        crop_bluriness = variance_of_laplacian(np.array(cropped_image))
        cropped_tensor = transform_species(cropped_image).unsqueeze(0).to(proc_device)

        class_name, class_confidence = classify_box(cropped_tensor, binary_model)
        order_name, order_confidence = classify_order(
            cropped_tensor, order_model, order_labels, order_data_thresholds
        )

        species_names, species_confidences, embedding = _get_species_and_embedding(
            class_name,
            order_name,
            cropped_tensor,
            regional_model,
            regional_category_map,
            top_n,
        )

        embedding_list[crop_status] = {
            "embedding": embedding,
            "image_path": os.path.basename(image_path),
            "image_size": [original_width, original_height],
            "crop": crop_status,
            "box": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max},
        }

        crop_path = ""
        if save_crops and crop_dir:
            crop_path = _save_crop_image(
                cropped_image, crop_dir, image_path, crop_status
            )

        best_matches = _get_best_matches(
            previous_image_embedding, crop_status, embedding_list
        )

        row = (
            [
                image_path,
                image_dt,
                *(_extract_deployment_fields(dep_data)),
                current_dt,
                recording_session,
                image_bluriness,
                crop_status,
                crop_bluriness,
                crop_area,
                crop_path,
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
            ]
            + species_names
            + species_confidences
            + [
                best_matches["previous_image"].values[0],
                best_matches["best_match_crop"].values[0],
                best_matches["cnn_cost"].values[0],
                best_matches["iou_cost"].values[0],
                best_matches["box_ratio_cost"].values[0],
                best_matches["dist_ratio_cost"].values[0],
                best_matches["total_cost"].values[0],
            ]
        )
        save_result_row(row, all_cols, csv_file)

    # append embedding to json
    if verbose:
        print(
            f" - Saving embedding for {len(embedding_list)} crops to {image_path.with_suffix('.json')}"
        )
    save_embedding(embedding_list, image_path, verbose=verbose)

    if skipped:
        row = _build_error_row(
            image_path,
            image_dt,
            dep_data,
            current_dt,
            recording_session,
            "No detections for this image.",
            all_cols,
        )
        save_result_row(row, all_cols, csv_file)


def download_and_analyse(
    keys,
    output_dir,
    dep_data,
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
    csv_file=Path("results.csv"),
    verbose=False,
):
    output_dir = Path(output_dir)
    csv_file = Path(csv_file)
    if verbose:
        print("Analysing images:")

    previous_image = None
    for key in keys:
        download_image_from_key(client, key, dep_data["country_code"], output_dir)
        local_path = output_dir / Path(key).name

        # Perform image analysis if enabled
        if perform_inference:
            perform_inf(
                local_path,
                dep_data=dep_data,
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
                verbose=verbose,
                previous_image=previous_image,
            )
            if previous_image is not None:
                prev_json = Path(previous_image).with_suffix(".json")
                if prev_json.is_file():
                    prev_json.unlink()
        previous_image = local_path
        if remove_image:
            local_path.unlink()
