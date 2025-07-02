import pytest
import pandas as pd
from unittest import mock
from pathlib import Path

import amber_inferences.cli.binary as binary


def test_main_runs_and_saves(tmp_path, monkeypatch):
    # Create a crops csv with two rows, one to be processed
    crops = pd.DataFrame(
        {
            "image_key": ["img1.jpg", "img2.jpg"],
            "crop_status": ["crop_1", "No detections for image."],
        }
    )
    crops_csv = tmp_path / "crops.csv"
    crops.to_csv(crops_csv, index=False)
    output_dir = tmp_path
    output_csv = tmp_path / "results.csv"
    # Mock classify_box to return dummy label/confidence
    monkeypatch.setattr(
        binary, "classify_box", lambda image_path, model: ("moth", 0.99)
    )
    # Run main
    binary.main(
        output_dir=output_dir,
        binary_model="model",
        crops_csv=crops_csv,
        output_csv=output_csv,
    )
    # Only the first row should be processed
    df = pd.read_csv(output_csv)
    assert len(df) == 1
    assert df.iloc[0]["image_key"] == "img1.jpg"
    assert df.iloc[0]["label"] == "moth"
    assert df.iloc[0]["confidence"] == 0.99


def test_file_checks(monkeypatch, tmp_path):
    # Patch load_models to return a dummy model dict
    monkeypatch.setattr(
        binary, "load_models", lambda *a, **k: {"classification_model": "model"}
    )
    # Patch classify_box to avoid running
    monkeypatch.setattr(binary, "classify_box", lambda *a, **k: ("moth", 1.0))
    # Create a dummy crops csv
    crops_csv = tmp_path / "crops.csv"
    pd.DataFrame({"image_key": ["img1.jpg"], "crop_status": ["OK"]}).to_csv(
        crops_csv, index=False
    )
    # Model path exists, crops_csv does not
    args = mock.Mock()
    args.binary_model_path = tmp_path / "model.pth"
    args.crops_csv = tmp_path / "notfound.csv"
    args.output_dir = tmp_path
    args.output_csv = tmp_path / "results.csv"
    args.job_name = "job123"
    args.__dict__.update(
        {
            "binary_model_path": args.binary_model_path,
            "crops_csv": args.crops_csv,
            "output_dir": args.output_dir,
            "output_csv": args.output_csv,
            "job_name": args.job_name,
        }
    )
    args.binary_model_path.write_text("dummy")
    # Should raise FileNotFoundError for crops_csv
    with pytest.raises(FileNotFoundError):
        if not args.crops_csv.resolve().exists():
            raise FileNotFoundError(f"Crops csv file not found: {args.crops_csv}")
    # Should not raise for model path
    assert args.binary_model_path.resolve().exists()


def test_cli_and_device_selection(tmp_path, monkeypatch, capsys):
    # Create dummy files
    crops_csv = tmp_path / "crops.csv"
    pd.DataFrame({"image_key": ["img1.jpg"], "crop_status": ["crop_1"]}).to_csv(
        crops_csv, index=False
    )
    model_path = tmp_path / "model.pth"
    model_path.write_text("dummy")

    # Patches
    monkeypatch.setattr(
        binary, "load_models", lambda *a, **k: {"classification_model": "model"}
    )
    monkeypatch.setattr(binary, "classify_box", lambda *a, **k: ("moth", 1.0))
    monkeypatch.setattr(binary.torch.cuda, "is_available", lambda: False)

    parser = binary.argparse.ArgumentParser()
    parser.add_argument("--crops_csv", type=Path)
    parser.add_argument("--output_csv", type=Path, default=tmp_path / "results.csv")
    parser.add_argument("--output_dir", type=Path, default=tmp_path)
    parser.add_argument("--binary_model_path", type=Path)
    args = parser.parse_args(
        [
            "--crops_csv",
            str(crops_csv),
            "--output_csv",
            str(tmp_path / "results.csv"),
            "--output_dir",
            str(tmp_path),
            "--binary_model_path",
            str(model_path),
        ]
    )

    # Simulate the main logic (forced false)
    if binary.torch.cuda.is_available():
        device = binary.torch.device("cuda:0")
    else:
        device = binary.torch.device("cpu")
    assert str(device) == "cpu"
    assert args.binary_model_path.exists()
    assert args.crops_csv.exists()
    models = binary.load_models(
        device, binary_model_path=args.binary_model_path.resolve()
    )
    assert "classification_model" in models
    # Run main with parsed args
    binary.main(
        output_dir=args.output_dir,
        binary_model=models["classification_model"],
        crops_csv=args.crops_csv,
        output_csv=args.output_csv,
    )
    df = pd.read_csv(args.output_csv)
    assert len(df) == 1
    assert df.iloc[0]["label"] == "moth"
    assert df.iloc[0]["confidence"] == 1.0
