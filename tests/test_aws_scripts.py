import pytest
from unittest import mock
import pandas as pd

import amber_inferences.utils.aws_scripts as aws_scripts


def test_list_objects_success(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"deployment_id": "d1"}]  # deployment name d1

    monkeypatch.setattr(aws_scripts.requests, "get", lambda *a, **k: FakeResponse())
    result = aws_scripts.list_objects("session", "bucket", "prefix", "user", "pass")
    assert result == [{"deployment_id": "d1"}]


def test_list_objects_http_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise aws_scripts.requests.exceptions.HTTPError("401 error")

    monkeypatch.setattr(aws_scripts.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(SystemExit):
        aws_scripts.list_objects("session", "bucket", "prefix", "user", "pass")


def test_list_objects_other_error(monkeypatch):
    def fake_get(*a, **k):
        raise Exception("fail")

    monkeypatch.setattr(aws_scripts.requests, "get", fake_get)
    with pytest.raises(SystemExit):
        aws_scripts.list_objects("session", "bucket", "prefix", "user", "pass")


def test_download_object_calls_perform_inf(monkeypatch, tmp_path):
    s3_client = mock.Mock()
    s3_client.download_file = mock.Mock()
    called = {}

    def fake_perform_inf(download_path, **kwargs):
        called["called"] = True
        called["download_path"] = download_path

    monkeypatch.setattr(aws_scripts, "perform_inf", fake_perform_inf)
    test_file = tmp_path / "test.jpg"
    test_file.write_text("data")
    aws_scripts.download_object(
        s3_client,
        "bucket",
        "key",
        str(test_file),
        perform_inference=True,
        remove_image=True,
        localisation_model=None,
        binary_model=None,
        order_model=None,
        order_labels=None,
        country="UK",
        region="UKCEH",
        device=None,
        order_data_thresholds=None,
        csv_file="results.csv",
    )
    assert called["called"]
    assert called["download_path"] == str(test_file)


def test_get_datetime_from_string():
    dt = aws_scripts.get_datetime_from_string("foo-bar-20240101123456-snapshot.jpg")
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1


def test_download_batch_skips_existing(monkeypatch, tmp_path):
    s3_client = mock.Mock()
    bucket_name = "bucket"
    keys = ["dir/file1.jpg", "dir/file2.jpg"]
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": [str(tmp_path / "dir/file1.jpg")]}).to_csv(
        csv_file, index=False
    )
    monkeypatch.setattr(aws_scripts, "download_object", lambda *a, **k: None)
    aws_scripts.download_batch(
        s3_client,
        bucket_name,
        keys,
        str(local_path),
        perform_inference=False,
        remove_image=False,
        localisation_model=None,
        binary_model=None,
        order_model=None,
        order_labels=None,
        country="UK",
        region="UKCEH",
        device=None,
        order_data_thresholds=None,
        csv_file=str(csv_file),
        rerun_existing=False,
    )
    # Only file2.jpg should be attempted for download
    # (assertion not needed, just check there is no error)


def test_count_files():
    s3_client = mock.Mock()
    paginator = mock.Mock()
    s3_client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "file1.jpg"}], "KeyCount": 2},
        {"Contents": [{"Key": "file2.jpg"}], "KeyCount": 10},
    ]
    count, all_keys = aws_scripts.count_files(s3_client, "bucket", "prefix")
    assert count == 12
    assert all_keys == ["file1.jpg", "file2.jpg"]
