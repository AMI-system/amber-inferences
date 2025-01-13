import os
import warnings
from datetime import datetime

import numpy as np
import torch

# import pandas as pd
# import torchvision.transforms as transforms
# from flat_bug.predictor import Predictor
# from PIL import Image

# ignore the pandas Future Warning
warnings.simplefilter(action="ignore", category=FutureWarning)


def flatbug(image_path, flatbug_model):
    print("Running flatbug")

    # Run inference on an image
    output = flatbug_model(image_path)

    print(len(output.json_data["boxes"]))
    # print(len(output.json_data.items().boxes))

    # Save a visualization of the predictions
    if len(output.json_data["boxes"]) > 0:
        print(f"Saving annotated image: {image_path}")
        output.plot(
            outpath=f"{os.path.dirname(image_path)}/flatbug/flatbug_{os.path.basename(image_path)}"
        )

    return output.json_data


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
    loc_model,
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
      - object classification
      - order classification
      - species classification
    """

    # transform_loc = transforms.Compose(
    #     [
    #         transforms.Resize((300, 300)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # transform_species = transforms.Compose(
    #     [
    #         transforms.Resize((300, 300)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )

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

    if not os.path.exists(f"{os.path.dirname(image_path)}/flatbug/"):
        os.makedirs(f"{os.path.dirname(image_path)}/flatbug/")
    flatbug(image_path, flatbug_model)
