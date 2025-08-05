import torch
import numpy as np
import pandas as pd
import unittest.mock as mock
import amber_inferences.utils.tracking as tracking
import math


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
            "best_match_crop": ["crop_a", "crop_b"],
            "image_path": ["b", "c"],
            "crop_status": ["crop_1", "crop_123"],
            "cnn_cost": [0.1, 0.2],
            "iou_cost": [0.1, 0.2],
            "box_ratio_cost": [0.1, 0.2],
            "dist_ratio_cost": [0.1, 0.2],
            "total_cost": [0.4, 0.8],
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1)
    assert "track_id" in out.columns
    assert out["track_id"].nunique() >= 1


def test_track_id_calc_first_last_included():
    # Simulate a sequence of 3 images with crops, so first and last are unique
    df = pd.DataFrame(
        {
            "previous_image": ["img1", "img2", "img2"],
            "best_match_crop": ["crop1", "crop2", "crop1"],
            "image_path": ["img2", "img3", "img3"],
            "crop_status": ["crop1", "crop1", "crop2"],
            "cnn_cost": [0.1, 0.2, 0.1],
            "iou_cost": [0.1, 0.2, 0.1],
            "box_ratio_cost": [0.1, 0.2, 0.1],
            "dist_ratio_cost": [0.1, 0.2, 0.1],
            "total_cost": [0.4, 0.8, 0.4],
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1)

    assert out["track_id"].nunique() == 2, "There should be 2 unique track IDs"
    assert (
        out.loc[
            (out["image_path"] == "img3") & (out["crop_id"] == "crop2"), "track_id"
        ].values[0]
        == out.loc[
            (out["image_path"] == "img2") & (out["crop_id"] == "crop1"), "track_id"
        ].values[0]
    )
    assert set(out["image_path"]) == set(
        list(set(df["image_path"])) + list(set(df["previous_image"]))
    )
    assert set(out["crop_id"]) == set(
        list(set(df["crop_status"])) + list(set(df["best_match_crop"]))
    )

    # check all df[['image_path', 'crop_status']] in out[['image_path', 'crop_id']]
    df_pairs = set(tuple(x) for x in df[["image_path", "crop_status"]].values)
    out_pairs = set(tuple(x) for x in out[["image_path", "crop_id"]].values)
    assert df_pairs.issubset(
        out_pairs
    ), "All (image_path, crop_status) pairs from df should be in out (image_path, crop_id)"

    df_pairs = set(tuple(x) for x in df[["previous_image", "best_match_crop"]].values)
    out_pairs = set(tuple(x) for x in out[["image_path", "crop_id"]].values)
    assert df_pairs.issubset(
        out_pairs
    ), "All (image_path, crop_status) pairs from df should be in out (image_path, crop_id)"


def test_extract_embedding(monkeypatch):
    import torch
    from PIL import Image
    import numpy as np

    # Dummy crop (PIL image)
    crop = Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 255)

    # Dummy model returns a tensor
    class DummyModel:
        def __call__(self, x):
            return torch.ones((1, 10, 1, 1))

    model = DummyModel()
    device = "cpu"
    # Patch transforms.Compose to identity
    monkeypatch.setattr(
        tracking,
        "transforms",
        mock.Mock(Compose=lambda x: lambda y: torch.ones((3, 300, 300))),
    )

    # Patch torch.no_grad to context manager
    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(tracking.torch, "no_grad", DummyNoGrad)
    # Patch l2_normalize to identity
    monkeypatch.setattr(tracking, "l2_normalize", lambda x: x)
    # Patch .to to identity
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: self)
    features = tracking.extract_embedding(crop, model, device)
    assert isinstance(features, np.ndarray) or isinstance(features, torch.Tensor)


def test_crop_costs(monkeypatch):
    # Setup dummy embedding_list
    embedding_list = {
        "img1": {
            "c1": {
                "embedding": np.array([1, 0]),
                "image_path": "img1",
                "crop": "c1",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            }
        },
        "img2": {
            "c2": {
                "embedding": np.array([1, 0]),
                "image_path": "img2",
                "crop": "c2",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            }
        },
    }
    # Patch calculate_cost to return a known DataFrame
    monkeypatch.setattr(
        tracking, "calculate_cost", lambda c_a, c_b: pd.DataFrame({"foo": [1]})
    )
    # Patch tqdm to identity
    monkeypatch.setattr(tracking, "tqdm", lambda x: x)
    df = tracking.crop_costs(embedding_list)
    assert isinstance(df, pd.DataFrame)
    assert "foo" in df.columns


def test_crop_costs_pair_generation(monkeypatch):
    # embedding_list with 2 images, 2 crops each
    embedding_list = {
        "img1": {
            "c1": {
                "embedding": np.array([1, 0]),
                "image_path": "img1",
                "crop": "c1",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            },
            "c2": {
                "embedding": np.array([0, 1]),
                "image_path": "img1",
                "crop": "c2",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            },
        },
        "img2": {
            "c3": {
                "embedding": np.array([1, 0]),
                "image_path": "img2",
                "crop": "c3",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            },
            "c4": {
                "embedding": np.array([0, 1]),
                "image_path": "img2",
                "crop": "c4",
                "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                "image_size": (10, 10),
            },
        },
    }
    call_args = []

    def fake_calculate_cost(c_a, c_b):
        call_args.append((c_a["crop"], c_b["crop"]))
        return pd.DataFrame({"foo": [1]})

    monkeypatch.setattr(tracking, "calculate_cost", fake_calculate_cost)
    monkeypatch.setattr(tracking, "tqdm", lambda x: x)
    tracking.crop_costs(embedding_list)
    # There should be 2*2 = 4 pairs between img1 and img2
    assert len(call_args) == 4
    expected_pairs = set((a, b) for a in ["c1", "c2"] for b in ["c3", "c4"])
    assert set(call_args) == expected_pairs


def test_crop_larger_threshold():
    # Test with a larger threshold
    df = pd.DataFrame(
        {
            "image_path": ["img1", "img2", "img3", "img3", "img4"],
            "crop_status": ["crop1", "crop1", "crop1", "crop2", "crop1"],
            "previous_image": [None, "img1", "img2", "img2", "img3"],
            "best_match_crop": [None, "crop1", "crop1", "crop1", "crop2"],
            "total_cost": [None, 0.8, 0.1, 1.9, 0.5],
        }
    )
    expected_out = pd.DataFrame(
        {
            "image_path": ["img1", "img2", "img3", "img3", "img4"],
            "crop_id": ["crop1", "crop1", "crop1", "crop2", "crop1"],
            "track_id": [
                "Track_00000",
                "Track_00000",
                "Track_00000",
                "Track_00001",
                "Track_00001",
            ],
            "total_cost": [math.inf, 0.8, 0.1, 1.9, 0.5],
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1)
    assert len(out["track_id"].values) == df.shape[0]
    assert all(out["image_path"].values == expected_out["image_path"].values)
    assert all(out["crop_id"].values == expected_out["crop_id"].values)
    assert all(out["track_id"].values == expected_out["track_id"].values)
    assert all(out["total_cost"].values == expected_out["total_cost"].values)


def test_exact_matches():
    # Test with a larger threshold
    df = pd.DataFrame(
        {
            "image_path": ["img1", "img2", "img3", "img3", "img3"],
            "crop_status": ["crop1", "crop1", "crop1", "crop2", "crop3"],
            "previous_image": [None, "img1", "img2", "img2", "img2"],
            "best_match_crop": [None, "crop1", "crop1", "crop1", "crop1"],
            "total_cost": [None, 0.8, 0.1, 0.05, 0.05],
        }
    )
    expected_out = pd.DataFrame(
        {
            "image_path": ["img1", "img2", "img3", "img3", "img3"],
            "crop_id": ["crop1", "crop1", "crop1", "crop2", "crop3"],
            "track_id": [
                "Track_00000",
                "Track_00000",
                "Track_00002",
                "Track_00000",
                "Track_00001",
            ],
            "total_cost": [math.inf, 0.8, 0.1, 0.05, 0.05],
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1)
    assert len(out["track_id"].values) == df.shape[0]
    assert all(out["image_path"].values == expected_out["image_path"].values)
    assert all(out["crop_id"].values == expected_out["crop_id"].values)
    assert set(out["track_id"].values) == set(expected_out["track_id"].values)
    assert all(out["total_cost"].values == expected_out["total_cost"].values)


def test_box_ratio_edge_cases():
    # Test with same size boxes
    bb1 = [0, 0, 10, 10]
    bb2 = [5, 5, 15, 15]
    ratio = tracking.box_ratio(bb1, bb2)
    assert ratio == 1.0  # Same area boxes

    # Test with very different sizes
    bb1 = [0, 0, 1, 1]
    bb2 = [0, 0, 10, 10]
    ratio = tracking.box_ratio(bb1, bb2)
    assert 0 < ratio < 1
    assert ratio < 0.5  # Much smaller box


def test_distance_ratio_edge_cases():
    # Test with same centers
    bb1 = [0, 0, 10, 10]
    bb2 = [0, 0, 20, 20]  # Different size, same center
    img_diag = 100
    ratio = tracking.distance_ratio(bb1, bb2, img_diag)
    true_val = 1 / (10 * math.sqrt(2))
    assert ratio == true_val

    # Test assertion error with distance > img_diag
    bb1 = [0, 0, 2, 2]
    bb2 = [100, 100, 102, 102]
    img_diag = 10
    try:
        tracking.distance_ratio(bb1, bb2, img_diag)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_cosine_similarity_edge_cases():
    # Test with zero vector
    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    sim = tracking.cosine_similarity(a, b)
    assert sim == 0.0

    # Test with both zero vectors
    a = np.array([0, 0])
    b = np.array([0, 0])
    sim = tracking.cosine_similarity(a, b)
    assert sim == 0.0

    # Test identical normalized vectors
    a = np.array([3, 4])
    b = np.array([6, 8])  # Same direction, different magnitude
    sim = tracking.cosine_similarity(a, b)
    assert np.isclose(sim, 1.0)


def test_iou_edge_cases():
    # Test perfect overlap
    bb1 = [0, 0, 10, 10]
    bb2 = [0, 0, 10, 10]
    iou_val = tracking.iou(bb1, bb2)
    assert np.isclose(iou_val, 1.0)

    # Test no overlap
    bb1 = [0, 0, 5, 5]
    bb2 = [10, 10, 15, 15]
    iou_val = tracking.iou(bb1, bb2)
    assert iou_val == 0.0

    # Test invalid bounding box assertion
    try:
        bb1 = [10, 0, 5, 10]  # xmin > xmax
        bb2 = [0, 0, 10, 10]
        tracking.iou(bb1, bb2)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


def test_l2_normalize_edge_cases():
    # Test unit vector
    t = torch.tensor([1.0, 0.0])
    normed = tracking.l2_normalize(t)
    assert torch.allclose(normed, torch.tensor([1.0, 0.0]))

    # Test negative values
    t = torch.tensor([-3.0, -4.0])
    normed = tracking.l2_normalize(t)
    assert np.isclose(normed.norm().item(), 1.0)


def test_calculate_cost_weights():
    # Test with custom weights
    crop1 = {
        "embedding": np.array([1, 0]),
        "image_path": "a",
        "crop": 0,
        "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "image_size": (10, 10),
    }
    crop2 = {
        "embedding": np.array([0, 1]),
        "image_path": "b",
        "crop": 1,
        "box": {"xmin": 2, "ymin": 2, "xmax": 3, "ymax": 3},
        "image_size": (10, 10),
    }

    # Test with different weights
    df = tracking.calculate_cost(crop1, crop2, w_cnn=2, w_iou=0.5, w_box=0.5, w_dis=1)
    assert "total_cost" in df.columns
    assert df["total_cost"].iloc[0] is not None

    # Verify the weighted sum is correct
    expected_total = (
        2 * df["cnn_cost"].iloc[0]
        + 0.5 * df["iou_cost"].iloc[0]
        + 0.5 * df["box_ratio_cost"].iloc[0]
        + 1 * df["dist_ratio_cost"].iloc[0]
    )
    assert np.isclose(df["total_cost"].iloc[0], expected_total)


def test_find_best_matches_edge_cases_one():
    # Test single row
    df = pd.DataFrame(
        {
            "crop1_path": ["a"],
            "crop1_crop": [0],
            "crop2_path": ["b"],
            "crop2_crop": [1],
            "cnn_cost": [0.1],
            "iou_cost": [0.2],
            "box_ratio_cost": [0.1],
            "dist_ratio_cost": [0.1],
            "total_cost": [0.5],
        }
    )
    best = tracking.find_best_matches(df)
    assert len(best) == 1
    assert best["total_cost"].iloc[0] == 0.5


def test_track_id_calc_high_cost_threshold():
    # Test that high costs break tracks
    df = pd.DataFrame(
        {
            "image_path": ["img2", "img3"],
            "crop_status": ["crop1", "crop1"],
            "previous_image": ["img1", "img2"],
            "best_match_crop": ["crop1", "crop1"],
            "total_cost": [5.0, 5.0],  # High costs
        }
    )
    out = tracking.track_id_calc(df, cost_threshold=1.0)
    # Should create separate tracks since costs exceed threshold
    track_ids = out["track_id"].unique()
    assert len(track_ids) >= 2


def test_extract_embedding_edge_cases(monkeypatch):
    from PIL import Image
    import numpy as np

    # Test with very small image
    crop = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 128)

    class DummyModel:
        def __call__(self, x):
            return torch.zeros((1, 5, 1, 1))  # Return zeros

    model = DummyModel()
    device = "cpu"

    # Mock transforms and other dependencies
    monkeypatch.setattr(
        tracking,
        "transforms",
        mock.Mock(Compose=lambda x: lambda y: torch.ones((3, 300, 300))),
    )

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(tracking.torch, "no_grad", DummyNoGrad)
    monkeypatch.setattr(tracking, "l2_normalize", lambda x: x)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: self)

    features = tracking.extract_embedding(crop, model, device)
    assert isinstance(features, (np.ndarray, torch.Tensor))
