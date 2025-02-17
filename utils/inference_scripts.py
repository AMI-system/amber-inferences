import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

# ignore the pandas Future Warning
warnings.simplefilter(action="ignore", category=FutureWarning)


def flatbug(image_path, flatbug_model):
    output = flatbug_model(image_path)

    # Save a visualization of the predictions
    # if len(output.json_data["boxes"]) > 0:
    #     print(f"Saving annotated image: {image_path}")
    #     output.plot(
    #         outpath=f"{os.path.dirname(image_path)}/flatbug/flatbug_{os.path.basename(image_path)}"
    #     )

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
    flatbug_model,
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
      - object detection
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
    image_dt = os.path.basename(image_path)

    if image_dt.startswith("20"):
        image_dt = image_dt.split("-")[0]
    else:
        image_dt = image_dt.split("-")[1]
    image_dt = datetime.strptime(image_dt, "%Y%m%d%H%M%S%f")
    image_dt = datetime.strftime(image_dt, "%Y-%m-%d %H:%M:%S")

    current_dt = datetime.now()
    current_dt = datetime.strftime(current_dt, "%Y-%m-%d %H:%M:%S")

    # if not os.path.exists(f"{os.path.dirname(image_path)}/flatbug/"):
    #     os.makedirs(f"{os.path.dirname(image_path)}/flatbug/")

    try:
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

    # # print('Inference for flatbug...')
    flatbug_outputs = flatbug(image_path, flatbug_model)

    original_image = image.copy()
    original_width, original_height = image.size

    skipped = []

    # catch no crops: if no boxes or all boxes below threshold
    if len(flatbug_outputs["boxes"]) == 0 or all(
        [score < box_threshold for score in flatbug_outputs["scores"]]
    ):
        skipped = [True]

    # for each detection
    for i in range(len(flatbug_outputs["boxes"])):
        crop_status = "crop " + str(i)

        x_min, y_min, x_max, y_max = flatbug_outputs["boxes"][i]

        box_score = flatbug_outputs["scores"][i]
        box_label = flatbug_outputs["labels"][i]

        x_min = int(int(x_min) * original_width / 300)
        y_min = int(int(y_min) * original_height / 300)
        x_max = int(int(x_max) * original_width / 300)
        y_max = int(int(y_max) * original_height / 300)

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
            species_names, species_confidences = classify_species(
                cropped_tensor, regional_model, regional_category_map, top_n
            )

        else:
            species_names, species_confidences = [""] * top_n, [""] * top_n

        # if save_crops then save the cropped image
        crop_path = ""
        if save_crops:
            crop_path = image_path.replace(".jpg", f"_crop{i}.jpg")
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
