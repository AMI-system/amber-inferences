#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracking utils for Amber Inferences
"""

import math
import numpy as np
from torchvision import transforms
import torch
from PIL import Image


def l2_normalize(tensor):
    norm = torch.norm(tensor, p=2)
    return tensor / norm if norm > 0 else tensor


def extract_embedding(crop, model, device):
    """Get the crop embedding"""
    embedding_transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    input_crop = embedding_transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(input_crop)

    # Flatten and normalize
    features = features.view(features.size(0), -1)  # shape: (1, N)
    features = features.squeeze(0).cpu()  # remove batch dim, move to CPU
    features = l2_normalize(features)  # normalize
    features = features.numpy()  # convert to numpy

    return features


def iou(bb1, bb2) -> float:
    """Finds intersection over union for a bounding box pair"""

    assert bb1[0] < bb1[2], f"Issue in bounding box 1 x_annotation: {bb1[0]} < {bb1[2]}"
    assert bb1[1] < bb1[3], f"Issue in bounding box 1 y_annotation: {bb1[1]} < {bb1[3]}"
    assert bb2[0] < bb2[2], f"Issue in bounding box 2 x_annotation: {bb2[0]} < {bb2[2]}"
    assert bb2[1] < bb2[3], f"Issue in bounding box 2 y_annotation: {bb2[1]} < {bb2[3]}"

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    x_min = max(bb1[0], bb2[0])
    x_max = min(bb1[2], bb2[2])
    width = max(0, x_max - x_min + 1)

    y_min = max(bb1[1], bb2[1])
    y_max = min(bb1[3], bb2[3])
    height = max(0, y_max - y_min + 1)

    intersec_area = width * height
    union_area = bb1_area + bb2_area - intersec_area

    iou = np.around(intersec_area / union_area, 2)
    assert 0 <= iou <= 1, "IoU out of bounds"

    return iou


def box_ratio(bb1, bb2) -> float:
    """Finds the ratio of the two bounding boxes"""

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    min_area = min(bb1_area, bb2_area)
    max_area = max(bb1_area, bb2_area)

    box_ratio = min_area / max_area
    assert 0 <= box_ratio <= 1, "box ratio out of bounds"

    return box_ratio


def distance_ratio(bb1, bb2, img_diag: float) -> float:
    """finds the distance between the two bounding boxes and normalizes
    by the image diagonal length
    """

    centre_x_bb1 = bb1[0] + (bb1[2] - bb1[0]) / 2
    centre_y_bb1 = bb1[1] + (bb1[3] - bb1[1]) / 2

    centre_x_bb2 = bb2[0] + (bb2[2] - bb2[0]) / 2
    centre_y_bb2 = bb2[1] + (bb2[3] - bb2[1]) / 2

    dist = math.sqrt(
        (centre_x_bb2 - centre_x_bb1) ** 2 + (centre_y_bb2 - centre_y_bb1) ** 2
    )
    max_dist = img_diag

    assert dist <= max_dist, "distance between bounding boxes more than max distance"

    return dist / max_dist


def cosine_similarity(img1_ftrs, img2_ftrs) -> float:
    """
    Finds cosine similarity between a pair of cropped images.

    Uses the feature embeddings array computed from a CNN model.
    """

    cosine_sim = np.dot(img1_ftrs, img2_ftrs) / (
        np.linalg.norm(img1_ftrs) * np.linalg.norm(img2_ftrs)
    )
    assert 0 <= cosine_sim <= 1.000000001, "Cosine similarity score out of bounds"

    return cosine_sim


def calculate_cost(crop1, crop2, w_cnn=1, w_iou=1, w_box=1, w_dis=1):
    """
    Calculate the cost between two crops based on their features and bounding boxes.
    """
    features1 = crop1["embedding"]
    features2 = crop2["embedding"]

    bb1 = crop1["box"]
    bb2 = crop2["box"]
    bb1 = [bb1["xmin"], bb1["ymin"], bb1["xmax"], bb1["ymax"]]
    bb2 = [bb2["xmin"], bb2["ymin"], bb2["xmax"], bb2["ymax"]]

    image = Image.open(crop1["image_path"]).convert("RGB")
    image_width, image_height = image.size

    diag = math.sqrt(image_width**2 + image_height**2)

    cnn_cost = 1 - cosine_similarity(features1, features2)
    iou_cost = 1 - iou(bb1, bb2)
    box_ratio_cost = 1 - box_ratio(bb1, bb2)
    dist_ratio_cost = distance_ratio(bb1, bb2, diag)

    total_cost = (
        w_cnn * cnn_cost
        + w_iou * iou_cost
        + w_box * box_ratio_cost
        + w_dis * dist_ratio_cost
    )

    return {
        "crop1_path": crop1["image_path"],
        "crop1_crop": crop1["crop"],
        "crop2_path": crop2["image_path"],
        "crop2_crop": crop2["crop"],
        "cnn_cost": cnn_cost,
        "iou_cost": iou_cost,
        "box_ratio_cost": box_ratio_cost,
        "dist_ratio_cost": dist_ratio_cost,
        "total_cost": total_cost,
    }
