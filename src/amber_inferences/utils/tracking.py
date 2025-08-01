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
    """
    Perform L2 normalization on a tensor.
    Args:
        tensor (torch.Tensor): Input tensor to normalize.
    Returns:
        torch.Tensor: L2-normalized tensor.
    """
    norm = torch.norm(tensor, p=2)
    return tensor / norm if norm > 0 else tensor


def extract_embedding(crop, model, device):
    """
    Extract a feature embedding from a crop using a model.
    Args:
        crop (PIL.Image.Image): Image crop to process.
        model (torch.nn.Module): Model to extract features.
        device (torch.device): Device to run the model on.
    Returns:
        np.ndarray: Normalized feature embedding as a numpy array.
    """
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
    """
    Compute the Intersection over Union (IoU) for a pair of bounding boxes.
    Args:
        bb1 (list or tuple): [xmin, ymin, xmax, ymax] for box 1.
        bb2 (list or tuple): [xmin, ymin, xmax, ymax] for box 2.
    Returns:
        float: IoU value between 0 and 1.
    Raises:
        AssertionError: If bounding box coordinates are invalid or IoU is out of bounds.
    """
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
    """
    Compute the ratio of the areas of two bounding boxes (min/max).
    Args:
        bb1 (list or tuple): [xmin, ymin, xmax, ymax] for box 1.
        bb2 (list or tuple): [xmin, ymin, xmax, ymax] for box 2.
    Returns:
        float: Area ratio between 0 and 1.
    Raises:
        AssertionError: If box ratio is out of bounds.
    """
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    min_area = min(bb1_area, bb2_area)
    max_area = max(bb1_area, bb2_area)

    box_ratio = min_area / max_area
    assert 0 <= box_ratio <= 1, "box ratio out of bounds"

    return box_ratio


def distance_ratio(bb1, bb2, img_diag: float) -> float:
    """
    Compute the normalized distance between the centers of two bounding boxes.
    Args:
        bb1 (list or tuple): [xmin, ymin, xmax, ymax] for box 1.
        bb2 (list or tuple): [xmin, ymin, xmax, ymax] for box 2.
        img_diag (float): Diagonal length of the image for normalization.
    Returns:
        float: Normalized distance ratio between 0 and 1.
    Raises:
        AssertionError: If distance is greater than the image diagonal.
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
    Compute the cosine similarity between two feature vectors.
    Args:
        img1_ftrs (np.ndarray): Feature vector for image 1.
        img2_ftrs (np.ndarray): Feature vector for image 2.
    Returns:
        float: Cosine similarity score between 0 and 1.
    Raises:
        AssertionError: If similarity is out of bounds.
    """
    # Check for zero norms first
    norm1 = np.linalg.norm(img1_ftrs)
    norm2 = np.linalg.norm(img2_ftrs)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_sim = np.dot(img1_ftrs, img2_ftrs) / (
        np.linalg.norm(img1_ftrs) * np.linalg.norm(img2_ftrs)
    )
    assert 0 <= cosine_sim <= 1.000000001, "Cosine similarity score out of bounds"

    return cosine_sim


def calculate_cost(crop1, crop2, w_cnn=1, w_iou=1, w_box=1, w_dis=1):
    """
    Calculate the cost between two crops based on their features and bounding boxes.
    Args:
        crop1 (dict): Crop dictionary with keys 'embedding', 'box', 'image_path', 'crop', 'image_size'.
        crop2 (dict): Crop dictionary with keys 'embedding', 'box', 'image_path', 'crop', 'image_size'.
        w_cnn (float): Weight for CNN feature cost.
        w_iou (float): Weight for IoU cost.
        w_box (float): Weight for box ratio cost.
        w_dis (float): Weight for distance ratio cost.
    Returns:
        pd.DataFrame: DataFrame with cost components and total cost for the crop pair.
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

    try:
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

    except Exception as e:
        print(f"Error calculating CNN cost: {e}")
        cnn_cost = None
        iou_cost = None
        box_ratio_cost = None
        dist_ratio_cost = None
        total_cost = None

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
    Find the best match and cost for each crop and those from the previous image.
    Args:
        df (pd.DataFrame): DataFrame with cost information for crop pairs.
    Returns:
        pd.DataFrame: DataFrame with best matches for each crop.
    """
    # Keep only best match for each (image, crop)
    filtered_df = df.copy()

    # check columns exist
    required_columns = [
        "crop1_path",
        "crop1_crop",
        "crop2_path",
        "crop2_crop",
        "cnn_cost",
        "iou_cost",
        "box_ratio_cost",
        "dist_ratio_cost",
        "total_cost",
    ]
    for col in required_columns:
        if col not in filtered_df.columns:
            raise ValueError(f"Missing required column: {col}")

    filtered_df.sort_values("total_cost", ascending=True, inplace=True)

    best_matches = filtered_df.drop_duplicates(
        subset=["crop1_path", "crop1_crop"], keep="first"
    )
    best_matches = best_matches.drop_duplicates(
        subset=["crop2_path", "crop2_crop"], keep="first"
    )
    if best_matches.shape[0] == 0:
        return pd.DataFrame(
            columns=[
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


def validate_input_columns(df, required_columns):
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


def preprocess_matches(df):
    df = df.copy()
    df["base_image_path"] = df["image_path"].apply(lambda x: os.path.basename(str(x)))
    df["image1"] = df["base_image_path"]
    df["image2"] = df["previous_image"]
    df["crop1"] = df["crop_status"]
    df["crop2"] = df["best_match_crop"]
    df.sort_values(by=["image1", "crop1"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["total_cost"] = df["total_cost"].astype(float)
    df = df[df["total_cost"].notna()]
    return df


def filter_matches(df, threshold):
    filtered = df[df["total_cost"] < threshold]
    return filtered.sort_values("total_cost").drop_duplicates(
        subset=["image2", "crop2"], keep="first"
    )


def build_match_graph(filtered_df):
    def node_id(image, crop):
        return f"{image}|{crop}"

    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        G.add_edge(
            node_id(row["image1"], row["crop1"]), node_id(row["image2"], row["crop2"])
        )
    return G


def assign_track_ids(graph):
    track_mapping = {}
    for tid, component in enumerate(nx.connected_components(graph)):
        for node in component:
            track_mapping[node] = f"Track_{str(tid).rjust(5, '0')}"
    return track_mapping


def build_cost_lookup(df):
    cost_lookup = {}
    for _, row in df.iterrows():
        nid = f"{row['image1']}|{row['crop1']}"
        cost = row["total_cost"]
        cost_lookup[nid] = min(cost_lookup.get(nid, float("inf")), cost)
    return cost_lookup


def generate_output(all_nodes, track_mapping, cost_lookup):
    output = []
    for node in all_nodes:
        image_path, crop_id = node.rsplit("|", maxsplit=1)
        output.append(
            {
                "image_path": image_path,
                "crop_id": crop_id,
                "track_id": track_mapping.get(node),
                "total_cost": cost_lookup.get(node, float("inf")),
            }
        )
    return pd.DataFrame(output)


def assign_unmatched_track_ids(df):
    max_id = max(
        [int(t.replace("Track_", "")) for t in df["track_id"].dropna()], default=-1
    )
    unmatched = df["track_id"].isnull()
    new_ids = [
        f"Track_{str(i).rjust(5, '0')}"
        for i in range(max_id + 1, max_id + 1 + unmatched.sum())
    ]
    df.loc[unmatched, "track_id"] = new_ids
    return df


def track_id_calc(best_matches, cost_threshold=1):
    required_columns = [
        "image_path",
        "crop_status",
        "previous_image",
        "best_match_crop",
        "total_cost",
    ]
    validate_input_columns(best_matches, required_columns)
    best_matches = preprocess_matches(best_matches)
    filtered_matches = filter_matches(best_matches, cost_threshold)
    G = build_match_graph(filtered_matches)
    track_mapping = assign_track_ids(G)

    # Gather all nodes
    all_nodes = set()
    for _, row in best_matches.iterrows():
        all_nodes.add(f"{row['image1']}|{row['crop1']}")
        all_nodes.add(f"{row['image2']}|{row['crop2']}")

    cost_lookup = build_cost_lookup(best_matches)
    output_df = generate_output(all_nodes, track_mapping, cost_lookup)
    output_df = assign_unmatched_track_ids(output_df)
    output_df.sort_values(by=["image_path", "crop_id"], inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def crop_costs(embedding_list):
    """
    Compute costs for all pairs of crops in consecutive images.
    Args:
        embedding_list (dict): Dictionary mapping image paths to crop dictionaries.
    Returns:
        pd.DataFrame: DataFrame with cost information for all crop pairs.
    """
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
