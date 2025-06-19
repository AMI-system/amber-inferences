import torch
import numpy as np
import pandas as pd

import amber_inferences.utils.tracking as tracking


def test_l2_normalize():
    t = torch.tensor([3.0, 4.0])
    normed = tracking.l2_normalize(t)
    np.testing.assert_allclose(normed.norm().item(), 1.0)
    t0 = torch.zeros(2)
    normed0 = tracking.l2_normalize(t0)
    assert torch.all(normed0 == 0)


def test_iou():
    bb1 = [0, 0, 10, 10]
    bb2 = [5, 5, 15, 15]
    iou_val = tracking.iou(bb1, bb2)
    assert 0 <= iou_val <= 1
    assert np.isclose(iou_val, 0.17, atol=0.01)


def test_box_ratio():
    bb1 = [0, 0, 10, 10]
    bb2 = [0, 0, 20, 20]
    ratio = tracking.box_ratio(bb1, bb2)
    assert 0 < ratio < 1

    # higher tol since area is nudged to allow for zero
    assert np.isclose(ratio, 0.25, atol=0.03)


def test_distance_ratio():
    bb1 = [0, 0, 10, 10]
    bb2 = [10, 10, 20, 20]
    img_diag = np.sqrt(100**2 + 100**2)
    ratio = tracking.distance_ratio(bb1, bb2, img_diag)
    assert 0 <= ratio <= 1


def test_cosine_similarity():
    a = np.array([1, 0])
    b = np.array([1, 0])
    sim = tracking.cosine_similarity(a, b)
    assert np.isclose(sim, 1.0)
    c = np.array([0, 1])
    sim2 = tracking.cosine_similarity(a, c)
    assert np.isclose(sim2, 0.0)


def test_calculate_cost_none():
    crop1 = {
        "embedding": None,
        "image_path": "a",
        "crop": 0,
        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "image_size": (10, 10),
    }
    crop2 = {
        "embedding": None,
        "image_path": "b",
        "crop": 1,
        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "image_size": (10, 10),
    }
    df = tracking.calculate_cost(crop1, crop2)
    assert df["cnn_cost"].iloc[0] is None


def test_calculate_cost_valid():
    crop1 = {
        "embedding": np.array([1, 0]),
        "image_path": "a",
        "crop": 0,
        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "image_size": (10, 10),
    }
    crop2 = {
        "embedding": np.array([1, 0]),
        "image_path": "b",
        "crop": 1,
        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "image_size": (10, 10),
    }
    df = tracking.calculate_cost(crop1, crop2)
    assert np.isclose(df["cnn_cost"].iloc[0], 0.0)
    assert 0 <= df["iou_cost"].iloc[0] <= 1
    assert 0 <= df["box_ratio_cost"].iloc[0] <= 1
    assert 0 <= df["dist_ratio_cost"].iloc[0] <= 1
    assert 0 <= df["total_cost"].iloc[0] <= 4


def test_find_best_matches():
    df = pd.DataFrame(
        {
            "crop1_path": ["a", "a", "b"],
            "crop1_crop": [0, 0, 1],
            "crop2_path": ["b", "c", "c"],
            "crop2_crop": [1, 2, 2],
            "cnn_cost": [0.1, 0.2, 0.3],
            "iou_cost": [0.1, 0.2, 0.3],
            "box_ratio_cost": [0.1, 0.2, 0.3],
            "dist_ratio_cost": [0.1, 0.2, 0.3],
            "total_cost": [0.4, 0.8, 0.9],
        }
    )
    best = tracking.find_best_matches(df)
    assert "track_id" not in best.columns
    assert best.shape[0] <= df.shape[0]


def test_track_id_calc():
    df = pd.DataFrame(
        {
            "previous_image": ["a", "b"],
            "best_match_crop": [0, 1],
            "image_path": ["b", "c"],
            "crop_status": [1, 2],
            "cnn_cost": [0.1, 0.2],
            "iou_cost": [0.1, 0.2],
            "box_ratio_cost": [0.1, 0.2],
            "dist_ratio_cost": [0.1, 0.2],
            "total_cost": [0.4, 0.8],
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1)
    assert "track_id" in out.columns
    assert "colour" in out.columns
    assert out["track_id"].nunique() >= 1
