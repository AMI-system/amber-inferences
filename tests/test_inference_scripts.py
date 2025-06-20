import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import torch

import amber_inferences.utils.inference_scripts as inference_scripts


# region Tests for variance_of_laplacian function
def test_variance_of_laplacian(monkeypatch):
    class DummyCV2:
        CV_64F = 0

        @staticmethod
        def Laplacian(image, flag):
            return np.ones((10, 10))

    monkeypatch.setattr(inference_scripts, "cv2", DummyCV2)
    image = np.ones((10, 10), dtype=np.uint8)
    var = inference_scripts.variance_of_laplacian(image)
    assert var == 0.0


@pytest.mark.parametrize(
    "image,expected",
    [
        (np.zeros((10, 10), dtype=np.uint8), 0.0),
        (np.ones((10, 10), dtype=np.uint8) * 255, 0.0),
    ],
)
def test_variance_of_laplacian_sharp(image, expected):
    assert inference_scripts.variance_of_laplacian(image) == expected


def test_variance_of_laplacian_blurr():
    image = np.eye(10, dtype=np.uint8) * 255
    result = inference_scripts.variance_of_laplacian(image)
    assert result > 1e4


def test_variance_of_laplacian_real(monkeypatch):
    image = np.ones((10, 10), dtype=np.uint8)
    result = inference_scripts.variance_of_laplacian(image)
    assert isinstance(result, float)


# endregion


# region Tests for get_image_metadata function
def test_get_image_metadata():
    path = Path("foo-20240101123456-bar.jpg")
    dt, session = inference_scripts.get_image_metadata(path)
    assert dt.startswith("2024-01-01")
    assert len(session) > 0


@pytest.mark.parametrize(
    "filename,expected_dt,expected_session",
    [
        ("foo-20240101123456-bar.jpg", "2024-01-01 12:34:56", "2024-01-01"),
        ("foo-20191231235959-bar.jpg", "2019-12-31 23:59:59", "2019-12-31"),
        ("foo-bar.jpg", "", ""),
    ],
)
def test_get_image_metadata_param(filename, expected_dt, expected_session):
    dt, session = inference_scripts.get_image_metadata(filename)
    assert dt == expected_dt
    assert session == expected_session


def test_get_image_metadata_invalid():
    dt, session = inference_scripts.get_image_metadata("not-a-date.jpg")
    assert dt == ""
    assert session == ""


def test_get_image_metadata_time_edge():
    # Test time before and after noon
    path = Path("foo-20240101115959-bar.jpg")
    dt, session = inference_scripts.get_image_metadata(path)
    assert session != ""
    path2 = Path("foo-20240101125959-bar.jpg")
    dt2, session2 = inference_scripts.get_image_metadata(path2)
    assert session2 != ""


# endregion


# region Tests for load_image function
def test_load_image_success(tmp_path):
    img_path = tmp_path / "img.jpg"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)
    out = inference_scripts.load_image(img_path)
    assert isinstance(out, Image.Image)


@pytest.mark.parametrize("img_exists", [True, False])
def test_load_image_param(tmp_path, img_exists):
    img_path = tmp_path / "img.jpg"
    if img_exists:
        img = Image.new("RGB", (10, 10))
        img.save(img_path)
        out = inference_scripts.load_image(img_path)
        assert isinstance(out, Image.Image)
    else:
        out = inference_scripts.load_image(img_path)
        assert out is None


def test_load_image_fail(tmp_path, capsys):
    out = inference_scripts.load_image(tmp_path / "notfound.jpg")
    assert out is None
    assert "Error opening image" in capsys.readouterr().out


def test_load_image_wrong_type():
    out = inference_scripts.load_image(12345)
    assert out is None


# endregion


# region Tests for initialise_session function
def test_initialise_session(tmp_path, monkeypatch):
    creds = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    cred_file = tmp_path / "creds.json"
    cred_file.write_text(json.dumps(creds))

    class DummySession:
        def client(self, *a, **k):
            return "client"

    monkeypatch.setattr("boto3.Session", lambda *a, **k: DummySession())
    client = inference_scripts.initialise_session(cred_file)
    assert client == "client"


# endregion


# region Tests for download_image_from_key function
def test_download_image_from_key(monkeypatch, tmp_path):
    class DummyClient:
        def download_file(self, bucket, key, path):
            Path(path).write_text("")

    output_dir = tmp_path / "out"
    inference_scripts.download_image_from_key(
        DummyClient(), "img1.jpg", "bucket", output_dir
    )
    assert (output_dir / "img1.jpg").exists()


# endregion


# region Tests for save_embedding function
def test_save_embedding(tmp_path):
    img_path = tmp_path / "img.jpg"
    data = {"a": np.array([1, 2, 3])}
    inference_scripts.save_embedding(data, img_path)
    json_path = img_path.with_suffix(".json")
    assert json_path.exists()


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"a": np.array([1, 2, 3])},
        {"b": [np.array([1, 2, 3]), np.array([4, 5, 6])]},
    ],
)
def test_save_embedding_edge(tmp_path, data):
    img_path = tmp_path / "img.jpg"
    inference_scripts.save_embedding(data, img_path)
    json_path = img_path.with_suffix(".json")
    assert json_path.exists()
    with open(json_path) as f:
        loaded = json.load(f)
    assert isinstance(loaded, dict)


def test_save_embedding_verbose(tmp_path, capsys):
    img_path = tmp_path / "img.jpg"
    data = {"a": np.array([1, 2, 3])}
    inference_scripts.save_embedding(data, img_path, verbose=True)
    assert "Saving embedding" in capsys.readouterr().out
    json_path = img_path.with_suffix(".json")
    assert json_path.exists()


# endregion


# region Tests for save_result_row function
def test_save_result_row(tmp_path):
    csv_file = tmp_path / "out.csv"
    data = [1, 2, 3]
    columns = ["a", "b", "c"]
    inference_scripts.save_result_row(data, columns, csv_file)
    df = pd.read_csv(csv_file)
    assert list(df.columns) == columns
    assert len(df) == 1


@pytest.mark.parametrize(
    "data,columns",
    [
        ([1, 2, 3], ["a", "b", "c"]),
        (["x", None, 3.5], ["col1", "col2", "col3"]),
    ],
)
def test_save_result_row_param(tmp_path, data, columns):
    csv_file = tmp_path / "out.csv"
    inference_scripts.save_result_row(data, columns, csv_file)
    df = pd.read_csv(csv_file)
    assert list(df.columns) == columns
    assert len(df) == 1


def test_save_result_row_multiple(tmp_path):
    csv_file = tmp_path / "multi.csv"
    data1 = [1, 2, 3]
    data2 = [4, 5, 6]
    columns = ["a", "b", "c"]
    inference_scripts.save_result_row(data1, columns, csv_file)
    inference_scripts.save_result_row(data2, columns, csv_file)
    df = pd.read_csv(csv_file)
    assert len(df) == 2
    assert list(df.columns) == columns


# endregion


# region Tests for get_previous_embedding function
def test_get_previous_embedding(tmp_path, capsys):
    img_path = tmp_path / "img.jpg"
    json_path = img_path.with_suffix(".json")
    with open(json_path, "w") as f:
        f.write("{}")
    out = inference_scripts.get_previous_embedding(img_path)
    assert isinstance(out, dict)
    out2 = inference_scripts.get_previous_embedding(None)
    assert isinstance(out2, dict)
    assert "No previous image embedding found" in capsys.readouterr().out


@pytest.mark.parametrize("img_exists", [True, False])
def test_get_previous_embedding_param(tmp_path, img_exists, capsys):
    img_path = tmp_path / "img.jpg"
    if img_exists:
        json_path = img_path.with_suffix(".json")
        with open(json_path, "w") as f:
            f.write("{}")
        out = inference_scripts.get_previous_embedding(img_path)
        assert isinstance(out, dict)
    else:
        out = inference_scripts.get_previous_embedding(img_path)
        assert isinstance(out, dict)
        assert "No previous image embedding found" in capsys.readouterr().out


def test_get_previous_embedding_no_json(tmp_path, capsys):
    img_path = tmp_path / "img.jpg"
    json_path = img_path.with_suffix(".json")
    json_path.write_text("not a json")
    out = inference_scripts.get_previous_embedding(img_path)
    assert isinstance(out, dict)
    assert out == {}


# endregion


# region Tests for get_default_row function
def test_get_default_row():
    row = inference_scripts.get_default_row(
        "img.jpg",
        "2024-01-01",
        "bucket",
        "2024-01-01",
        "2024-01-01",
        0.1,
        "msg",
        ["a"] * 10,
    )
    assert len(row) == 10


# endregion


# region Tests for convert_ndarrays function
def test_convert_ndarrays():
    arr = np.array([1, 2, 3])
    d = {"a": arr, "b": [arr, arr]}
    out = inference_scripts.convert_ndarrays(d)
    assert isinstance(out["a"], list)
    assert isinstance(out["b"][0], list)
    assert isinstance(inference_scripts.convert_ndarrays(arr), list)
    assert isinstance(inference_scripts.convert_ndarrays({"a": arr}), dict)
    assert isinstance(inference_scripts.convert_ndarrays([arr]), list)
    assert inference_scripts.convert_ndarrays("foo") == "foo"


@pytest.mark.parametrize(
    "obj,expected_type",
    [
        ({"a": np.array([1, 2, 3])}, dict),
        ([np.array([1, 2, 3]), np.array([4, 5, 6])], list),
        (np.array([1, 2, 3]), list),
        ("string", str),
    ],
)
def test_convert_ndarrays_param(obj, expected_type):
    out = inference_scripts.convert_ndarrays(obj)
    assert isinstance(out, expected_type)


def test_convert_ndarrays_nested():
    arr = np.array([1, 2, 3])
    obj = {"a": [arr, {"b": arr}]}
    out = inference_scripts.convert_ndarrays(obj)
    assert isinstance(out["a"], list)
    assert isinstance(out["a"][1]["b"], list)


# endregion


# region Tests crop_image_only function
def test_crop_image_only_error(tmp_path):
    # Should handle missing image gracefully
    csv_file = tmp_path / "out.csv"
    result = inference_scripts.crop_image_only(
        image_path=tmp_path / "notfound.jpg",
        bucket_name="bucket",
        localisation_model=None,
        proc_device="cpu",
        csv_file=csv_file,
        save_crops=False,
    )
    assert result is None
    df = pd.read_csv(csv_file)
    assert "Image corrupt" in df.values


# endregion


# region Test classify_box function (binary inference)
def test_classify_box_param():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Simulate logits for two classes: moth and nonmoth
            return torch.tensor([[0.9, 0.1]])  # logits, not softmaxed

    # Dummy input tensor (correct shape expected by model)
    tensor = torch.zeros((1, 3, 300, 300))

    # Call your function
    label, score = inference_scripts.classify_box(tensor, DummyModel())

    # Validate
    assert label in ["moth", "nonmoth"]
    assert isinstance(score, float)
    assert label == "moth"
    assert 0.0 <= score <= 1.0


def test_classify_box(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9]])

    tensor = torch.zeros((1, 3, 300, 300))
    label, score = inference_scripts.classify_box(tensor, DummyModel())
    assert label in ["moth", "nonmoth"]
    assert isinstance(score, float)


# endregion


# region Tests get_default_row function
def test_get_default_row_edge():
    row = inference_scripts.get_default_row(
        "img.jpg", "", "", "", "", None, None, ["a"] * 8
    )
    assert len(row) == 8


# endregion


# region Tests the species inference pipeline
def test_classify_species():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9, 0.0]])  # logits for 3 classes

    # Dummy input image tensor
    image_tensor = torch.zeros((1, 3, 300, 300))

    # Dummy category map (index -> label)
    regional_category_map = {"a": 0, "b": 1, "c": 2}

    labels, scores, features = inference_scripts.classify_species(
        image_tensor, DummyModel(), regional_category_map, top_n=2
    )

    assert isinstance(labels, list)
    assert isinstance(scores, list)
    assert len(labels) == 2
    assert all(isinstance(lab, str) for lab in labels)
    assert all(isinstance(sc, np.float32) for sc in scores)
    assert isinstance(features, np.ndarray)


def test_classify_species_topn(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9, 0.0]])

    image_tensor = torch.zeros((1, 3, 300, 300))
    regional_category_map = {"a": 0, "b": 1, "c": 2}
    labels, scores, features = inference_scripts.classify_species(
        image_tensor, DummyModel(), regional_category_map, top_n=3
    )
    assert len(labels) == 3
    assert all(isinstance(lab, str) for lab in labels)
    assert all(isinstance(s, np.float32) for s in scores)
    assert isinstance(features, np.ndarray)


# endregion


# region Test the order pipeline
def test_classify_order():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9, 0.0]])

    image_tensor = torch.zeros((1, 3, 300, 300))
    order_labels = {0: "Coleoptera", 1: "Lepidoptera", 2: "Diptera"}
    order_data_thresholds = {}  # ignored in current code

    label, score = inference_scripts.classify_order(
        image_tensor, DummyModel(), order_labels, order_data_thresholds
    )

    assert label in order_labels.values()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# endregion


# region Test the get_boxes function with various models
def test_get_boxes_flatbug(monkeypatch):
    # Mock the flatbug model and inference function
    class DummyFlatbugModel:
        def __call__(self, image_path):
            class Output:
                json_data = {"boxes": [[1, 2, 3, 4]], "confs": [0.9], "classes": [1]}

                def plot(self, outpath):
                    pass

            return Output()

    monkeypatch.setattr(
        inference_scripts,
        "flatbug",
        lambda image_path, model: {
            "boxes": [[1, 2, 3, 4]],
            "scores": [0.9],
            "labels": [1],
        },
    )
    out = inference_scripts.get_boxes("notfasterrcnn", None, "img.jpg", 300, 300, "cpu")
    assert isinstance(out, list)
    assert "boxes" in out[0] or isinstance(out[1], list)


def test_get_boxes_fasterrcnn(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.__class__.__name__ = "FasterRCNN"

        def __call__(self, x):
            return [
                {
                    "boxes": [torch.tensor([0, 0, 10, 10])],
                    "scores": [0.1],
                    "labels": [1],
                }
            ]

    image = Image.new("RGB", (300, 300))
    out = inference_scripts.get_boxes(DummyModel(), image, "img.jpg", 300, 300, "cpu")
    assert isinstance(out, list)
    assert isinstance(out[0], dict)
    assert isinstance(out[1], list)


# endregion


# region Test the flatbug model inference
def test_flatbug(monkeypatch, tmp_path):
    class DummyFlatbugModel:
        def __call__(self, image_path):
            class Output:
                json_data = {"boxes": [[1, 2, 3, 4]], "confs": [0.9], "classes": [1]}

                def plot(self, outpath):
                    pass

            return Output()

    img_path = tmp_path / "img.jpg"
    img_path.write_text("")
    out = inference_scripts.flatbug(
        str(img_path), DummyFlatbugModel(), save_annotation=True
    )
    assert "boxes" in out and "scores" in out and "labels" in out


# endregion


# region Test crop_image_only function
def test_crop_image_only_all_skipped(tmp_path, monkeypatch):
    # All boxes below threshold
    class DummyModel:
        def __init__(self):
            self.__class__.__name__ = "FasterRCNN"

        def __call__(self, x):
            return [
                {
                    "boxes": [torch.tensor([0, 0, 10, 10])],
                    "scores": [0.1],
                    "labels": [1],
                }
            ]

    img_path = tmp_path / "img.jpg"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)
    csv_file = tmp_path / "out.csv"
    _ = inference_scripts.crop_image_only(
        image_path=img_path,
        bucket_name="bucket",
        localisation_model=DummyModel(),
        proc_device="cpu",
        csv_file=csv_file,
        save_crops=False,
        box_threshold=0.99,
    )
    # check the data was output correctly and no crops detected
    df = pd.read_csv(csv_file)
    assert "No detections for image." in df.values


# endregion


# region Test localisation_only function
def test_localisation_only(monkeypatch, tmp_path):
    # Patch S3 client and crop_image_only
    class DummyClient:
        def download_file(self, bucket, key, path, Config=None):
            Path(path).write_text("")

    monkeypatch.setattr(inference_scripts, "crop_image_only", lambda *a, **k: None)
    keys = ["img1.jpg", "img2.jpg"]
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv_file = tmp_path / "results.csv"
    inference_scripts.localisation_only(
        keys,
        output_dir,
        bucket_name="bucket",
        client=DummyClient(),
        remove_image=True,
        perform_inference=True,
        save_crops=False,
        localisation_model=None,
        box_threshold=0.99,
        device="cpu",
        csv_file=csv_file,
        job_name="job1",
    )
    # Should remove files after download
    assert not any((output_dir / "snapshots").glob("*.jpg"))


# endregion


# region Integration Tests
def test_save_and_load_embedding_integration(tmp_path):
    img_path = tmp_path / "img.jpg"
    data = {"a": np.array([1, 2, 3])}
    # Save embedding
    inference_scripts.save_embedding(data, img_path)
    # Load embedding
    loaded = inference_scripts.get_previous_embedding(img_path)
    assert isinstance(loaded, dict)
    assert "a" in loaded
    assert isinstance(loaded["a"], list)


def test_save_and_load_image_integration(tmp_path):
    img_path = tmp_path / "img.jpg"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)
    # Load image
    loaded_img = inference_scripts.load_image(img_path)
    assert isinstance(loaded_img, Image.Image)
    # Save embedding for this image
    data = {"shape": list(loaded_img.size)}
    inference_scripts.save_embedding(data, img_path)
    loaded = inference_scripts.get_previous_embedding(img_path)
    assert loaded["shape"] == [10, 10]


def test_save_and_append_result_row_integration(tmp_path):
    csv_file = tmp_path / "results.csv"
    columns = ["a", "b", "c"]
    row1 = [1, 2, 3]
    row2 = [4, 5, 6]
    inference_scripts.save_result_row(row1, columns, csv_file)
    inference_scripts.save_result_row(row2, columns, csv_file)
    df = pd.read_csv(csv_file)
    assert len(df) == 2
    assert list(df.columns) == columns
    assert df.iloc[0]["a"] == 1
    assert df.iloc[1]["b"] == 5


def test_perform_inf_integration(monkeypatch, tmp_path):
    # Mock all model calls and dependencies
    monkeypatch.setattr(
        inference_scripts,
        "get_boxes",
        lambda *a, **k: ({"scores": [1.0], "labels": [0]}, [[0, 0, 10, 10]]),
    )
    monkeypatch.setattr(
        inference_scripts, "classify_box", lambda *a, **k: ("moth", 0.99)
    )
    monkeypatch.setattr(
        inference_scripts, "classify_order", lambda *a, **k: ("Lepidoptera", 0.95)
    )
    monkeypatch.setattr(
        inference_scripts,
        "classify_species",
        lambda *a, **k: (["species1"], [0.9], [1, 2, 3]),
    )
    monkeypatch.setattr(
        inference_scripts,
        "calculate_cost",
        lambda *a, **k: pd.DataFrame(
            {
                "cnn_cost": [0],
                "iou_cost": [0],
                "box_ratio_cost": [0],
                "dist_ratio_cost": [0],
                "total_cost": [0],
                "previous_image": [None],
                "best_match_crop": [None],
            }
        ),
    )
    monkeypatch.setattr(
        inference_scripts,
        "find_best_matches",
        lambda x: pd.DataFrame(
            {
                "previous_image": [None],
                "best_match_crop": [None],
                "cnn_cost": [0],
                "iou_cost": [0],
                "box_ratio_cost": [0],
                "dist_ratio_cost": [0],
                "total_cost": [0],
            }
        ),
    )
    img_path = tmp_path / "img-20240101123456-test.jpg"
    img = Image.new("RGB", (10, 10))
    img.save(img_path)
    csv_file = tmp_path / "out.csv"
    inference_scripts.perform_inf(
        image_path=img_path,
        bucket_name="bucket",
        localisation_model=None,
        binary_model=None,
        order_model=None,
        order_labels={0: "Lepidoptera"},
        regional_model=None,
        regional_category_map={"species1": 0},
        proc_device="cpu",
        order_data_thresholds={},
        csv_file=csv_file,
        save_crops=False,
        box_threshold=0.5,
        top_n=1,
        verbose=True,
        previous_image=None,
    )
    df = pd.read_csv(csv_file)
    assert "image_path" in df.columns
    assert len(df) == 1


# endregion
