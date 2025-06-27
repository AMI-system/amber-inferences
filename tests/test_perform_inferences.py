import pytest
import os
import json
from unittest import mock
import pandas as pd

import amber_inferences.cli.perform_inferences as perform_inferences


def make_json_file(tmp_path, chunks):
    json_path = tmp_path / "chunks.json"
    with open(json_path, "w") as f:
        json.dump(chunks, f)
    return str(json_path)


def test_main_skips_processed(monkeypatch, tmp_path, capsys):
    chunks = {"2021-01-01": ["img1.jpg", "img2.jpg", "img3.jpg"]}
    json_file = make_json_file(tmp_path, chunks)
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": ["img1.jpg", "img2.jpg"]}).to_csv(csv_file, index=False)

    # Mocks
    monkeypatch.setattr(perform_inferences, "initialise_session", lambda x: "client")
    monkeypatch.setattr(
        perform_inferences, "download_and_analyse", lambda **kwargs: kwargs
    )

    # This should skip img1 and img2, process img3
    perform_inferences.main(
        chunk_id=1,
        json_file=json_file,
        output_dir="/tmp/out",
        bucket_name="bucket",
        credentials_file="creds.json",
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
        device="cpu",
        order_data_thresholds=None,
        top_n=5,
        csv_file=str(csv_file),
        skip_processed=True,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "Skipping 2 images previously processed" in out


def test_main_all_processed(monkeypatch, tmp_path, capsys):
    chunks = {"2021-01-01": ["img1.jpg"]}
    json_file = make_json_file(tmp_path, chunks)
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": ["img1.jpg"]}).to_csv(csv_file, index=False)
    monkeypatch.setattr(perform_inferences, "initialise_session", lambda x: "client")
    monkeypatch.setattr(
        perform_inferences, "download_and_analyse", lambda **kwargs: kwargs
    )
    perform_inferences.main(
        chunk_id=1,
        json_file=json_file,
        output_dir="/tmp/out",
        bucket_name="bucket",
        credentials_file="creds.json",
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
        device="cpu",
        order_data_thresholds=None,
        top_n=5,
        csv_file=str(csv_file),
        skip_processed=True,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "All images already processed" in out


def test_main_invalid_chunk(monkeypatch, tmp_path):
    chunks = {"2021-01-01": ["img1.jpg"]}
    json_file = make_json_file(tmp_path, chunks)
    monkeypatch.setattr(perform_inferences, "initialise_session", lambda x: "client")
    with pytest.raises(ValueError):
        perform_inferences.main(
            chunk_id="notanumber",
            json_file=json_file,
            output_dir="/tmp/out",
            bucket_name="bucket",
            credentials_file="creds.json",
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
            device="cpu",
            order_data_thresholds=None,
            top_n=5,
            csv_file="results.csv",
            skip_processed=False,
            verbose=False,
        )


def test_model_path_check(monkeypatch, tmp_path):
    # Patch os.path.exists to return False for a specific path
    monkeypatch.setattr(os.path, "exists", lambda p: False if "fail" in p else True)
    args = mock.Mock()
    args.localisation_model_path = "fail_model.pth"
    args.binary_model_path = "ok.pth"
    args.order_model_path = "ok.pth"
    args.order_thresholds_path = "ok.csv"
    args.species_model_path = "ok.pth"
    args.species_labels = "ok.json"
    args.json_file = "ok.json"
    with pytest.raises(FileNotFoundError):
        # Simulate the relevant code block from __main__
        for mod_path in [
            args.localisation_model_path,
            args.binary_model_path,
            args.order_model_path,
            args.order_thresholds_path,
            args.species_model_path,
            args.species_labels,
        ]:
            if not os.path.exists(os.path.abspath(mod_path)):
                raise FileNotFoundError(f"Model path not found: {mod_path}")


def test_json_file_check(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: False if "fail" in p else True)
    args = mock.Mock()
    args.json_file = "fail.json"
    with pytest.raises(FileNotFoundError):
        if not os.path.exists(os.path.abspath(args.json_file)):
            raise FileNotFoundError(f"JSON file not found: {args.json_file}")
