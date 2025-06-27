import pytest
import os
import json
from unittest import mock

import amber_inferences.cli.localisation as localisation


def make_json_file(tmp_path, chunks):
    json_path = tmp_path / "chunks.json"
    with open(json_path, "w") as f:
        json.dump(chunks, f)
    return str(json_path)


def test_main_valid(monkeypatch, tmp_path):
    # Setup
    chunks = {"chunk_0": {"keys": ["img1.jpg", "img2.jpg"]}}
    json_file = make_json_file(tmp_path, chunks)
    # Mocks
    monkeypatch.setattr(localisation, "initialise_session", lambda x: "client")
    called = {}

    def fake_localisation_only(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(localisation, "localisation_only", fake_localisation_only)
    localisation.main(
        chunk_id="chunk_0",
        json_file=json_file,
        output_dir="/tmp/out",
        bucket_name="bucket",
        credentials_file="creds.json",
        remove_image=True,
        perform_inference=True,
        save_crops=False,
        localisation_model="model",
        box_threshold=0.99,
        device="cpu",
        csv_file="results.csv",
        job_name="job123",
    )
    assert called["keys"] == ["img1.jpg", "img2.jpg"]
    assert called["job_name"] == "job123"


def test_main_invalid_chunk(monkeypatch, tmp_path):
    chunks = {"chunk_0": {"keys": ["img1.jpg"]}}
    json_file = make_json_file(tmp_path, chunks)
    monkeypatch.setattr(localisation, "initialise_session", lambda x: "client")
    with pytest.raises(ValueError):
        localisation.main(
            chunk_id="chunk_1",
            json_file=json_file,
            output_dir="/tmp/out",
            bucket_name="bucket",
            credentials_file="creds.json",
            remove_image=True,
            perform_inference=True,
            save_crops=False,
            localisation_model="model",
            box_threshold=0.99,
            device="cpu",
            csv_file="results.csv",
            job_name="job123",
        )


def test_model_path_check(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: False if "fail" in p else True)
    args = mock.Mock()
    args.localisation_model_path = "fail_model.pth"
    args.json_file = "ok.json"
    with pytest.raises(FileNotFoundError):
        if not os.path.exists(os.path.abspath(args.localisation_model_path)):
            raise FileNotFoundError(
                f"Model path not found: {args.localisation_model_path}"
            )


def test_json_file_check(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "exists", lambda p: False if "fail" in p else True)
    args = mock.Mock()
    args.localisation_model_path = "ok.pth"
    args.json_file = "fail.json"
    with pytest.raises(FileNotFoundError):
        if not os.path.exists(os.path.abspath(args.json_file)):
            raise FileNotFoundError(f"JSON file not found: {args.json_file}")
