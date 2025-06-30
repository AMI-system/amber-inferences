import math
import numpy as np
from torchvision import transforms
import torch
import pandas as pd
import os
import networkx as nx
from itertools import product
from tqdm import tqdm


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

    if features1 is None or features2 is None:
        return pd.DataFrame(
            {
                "crop1_path": [crop1["image_path"]],
                "crop1_crop": [crop1["crop"]],
                "crop2_path": [crop2["image_path"]],
                "crop2_crop": [crop2["crop"]],
                "cnn_cost": [None],
                "iou_cost": [None],
                "box_ratio_cost": [None],
                "dist_ratio_cost": [None],
                "total_cost": [None],
            }
        ).reset_index(drop=True)

    bb1 = crop1["box"]
    bb2 = crop2["box"]
    bb1 = [bb1["xmin"], bb1["ymin"], bb1["xmax"], bb1["ymax"]]
    bb2 = [bb2["xmin"], bb2["ymin"], bb2["xmax"], bb2["ymax"]]

    image_width, image_height = crop1["image_size"]

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

    results = {
        "crop1_path": [crop1["image_path"]],
        "crop1_crop": [crop1["crop"]],
        "crop2_path": [crop2["image_path"]],
        "crop2_crop": [crop2["crop"]],
        "cnn_cost": [cnn_cost],
        "iou_cost": [iou_cost],
        "box_ratio_cost": [box_ratio_cost],
        "dist_ratio_cost": [dist_ratio_cost],
        "total_cost": [total_cost],
    }
    results_df = pd.DataFrame(results).reset_index(drop=True)

    return results_df


def find_best_matches(df):
    """
    Calculate the best match and cost for a crop and those from the previous image
    """
    # Keep only best match for each (image, crop)
    filtered_df = df.copy()
    filtered_df.sort_values("total_cost", ascending=True, inplace=True)

    best_matches = filtered_df.drop_duplicates(
        subset=["crop1_path", "crop1_crop"], keep="first"
    )
    best_matches = best_matches.drop_duplicates(
        subset=["crop2_path", "crop2_crop"], keep="first"
    )
    best_matches.columns = [
        "previous_image",
        "best_match_crop",
        "image_path",
        "crop_status",
        "cnn_cost",
        "iou_cost",
        "box_ratio_cost",
        "dist_ratio_cost",
        "total_cost",
    ]
    return best_matches


def track_id_calc(best_matches, cost_threshold=1):
    # Make a copy and rename the relevant columns
    best_matches = best_matches.copy()
    best_matches["base_image_path"] = best_matches["image_path"].apply(
        lambda x: os.path.dirname(x)
    )

    best_matches["image1"] = best_matches["base_image_path"]
    best_matches["image2"] = best_matches["previous_image"]
    best_matches["crop1"] = best_matches["crop_status"]
    best_matches["crop2"] = best_matches["best_match_crop"]

    # dummy populate the first frame values, in order for tracks to start there
    # this works by tracking these crops to themselves
    first_frame = best_matches[
        best_matches["image1"] == best_matches["image1"].values[0]
    ].copy()

    best_matches = best_matches.loc[best_matches["total_cost"] != "",]
    best_matches = best_matches.loc[best_matches["total_cost"].notna(),]

    first_frame = first_frame[
        (first_frame["image1"] + first_frame["crop1"]).isin(
            best_matches["image2"] + best_matches["crop2"]
        )
    ]
    first_frame["image2"] = first_frame["image1"]
    first_frame["crop2"] = first_frame["crop1"]
    first_frame["total_cost"] = 0

    best_matches = pd.concat([best_matches, first_frame], ignore_index=True)
    best_matches = best_matches.loc[best_matches["total_cost"] != "",]
    best_matches = best_matches.loc[best_matches["total_cost"].notna(),]

    # Filter based on the cost threshold
    filtered_matches = best_matches[best_matches["total_cost"] < cost_threshold]

    def node_id(image_path, crop_id):
        return f"{image_path}|{crop_id}"

    # Build a graph of matches under the threshold
    G_thresh = nx.Graph()
    for _, row in filtered_matches.iterrows():
        n1 = node_id(row["image1"], row["crop1"])
        n2 = node_id(row["image2"], row["crop2"])
        G_thresh.add_edge(n1, n2)

    # Assign track IDs from connected components
    track_mapping = {}
    for tid, component in enumerate(nx.connected_components(G_thresh)):
        for node in component:
            track_mapping[node] = f"Track_{str(tid).rjust(5, '0')}"

    # Collect all unique nodes from both sides
    all_nodes = set()
    for _, row in best_matches.iterrows():
        all_nodes.add(node_id(row["image1"], row["crop1"]))
        all_nodes.add(node_id(row["image2"], row["crop2"]))

    # Create lookup for cost (minimum per crop node)
    cost_lookup = {}
    for _, row in best_matches.iterrows():
        for prefix in ["1", "2"]:
            nid = node_id(row[f"image{prefix}"], row[f"crop{prefix}"])
            cost = row["total_cost"]
            cost_lookup[nid] = min(cost_lookup.get(nid, float("inf")), cost)

    # Assemble the final output rows
    output_rows = []
    for node in all_nodes:
        image_path, crop_id = node.rsplit("|", 2)
        output_rows.append(
            {
                "image_path": image_path,
                "crop_id": crop_id,
                "track_id": track_mapping.get(
                    node
                ),  # May be None if unmatched under threshold
                "total_cost": cost_lookup.get(node),
            }
        )

    output_df = pd.DataFrame(output_rows)

    # Assign unique track IDs to unmatched crops
    max_existing_id = max(
        [int(tid.replace("Track_", "")) for tid in output_df["track_id"].dropna()],
        default=-1,
    )
    unmatched_mask = output_df["track_id"].isnull()
    unmatched_indices = output_df[unmatched_mask].index
    new_ids = [
        f"Track_{str(i).rjust(5, '0')}"
        for i in range(max_existing_id + 1, max_existing_id + 1 + unmatched_mask.sum())
    ]
    output_df.loc[unmatched_indices, "track_id"] = new_ids

    return output_df


def crop_costs(embedding_list):
    all_crop_pairs = []
    image_paths = list(embedding_list.keys())

    for i in range(len(image_paths) - 1):
        img1 = image_paths[i]
        img2 = image_paths[i + 1]

        crops1 = embedding_list[img1]
        crops2 = embedding_list[img2]

        for c1, c2 in product(crops1, crops2):
            all_crop_pairs.append((img1, c1, img2, c2))

    for image_a, crop_a, image_b, crop_b in tqdm(all_crop_pairs):
        c_a = embedding_list[image_a][crop_a]
        c_a["image_path"] = image_a
        c_b = embedding_list[image_b][crop_b]
        c_b["image_path"] = image_b

        results_df = calculate_cost(c_a, c_b)

    return results_df
