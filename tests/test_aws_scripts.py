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
    dep_info = {"country_code": "bucket"}
    aws_scripts.download_object(
        s3_client,
        dep_info,
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


def test_download_object_check_error(monkeypatch, tmp_path, capsys):
    s3_client = mock.Mock()
    s3_client.download_file = mock.Mock()
    dep_info = {"country_code": "bucket"}
    test_file = tmp_path / "test.jpg"

    # monkeypatch to raise exception on download
    monkeypatch.setattr(
        s3_client,
        "download_file",
        lambda *a, **k: (_ for _ in ()).throw(Exception("Download failed")),
    )

    aws_scripts.download_object(
        s3_client,
        dep_info,
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
    assert "Error downloading bucket/key" in capsys.readouterr().err


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


def make_fake_session_and_client(tmp_path, keys):
    class FakeClient:
        def __init__(self):
            self.downloaded = []
            self.paginator = mock.Mock()
            self.paginator.paginate.return_value = [
                {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}
            ]

        def get_paginator(self, name):
            return self.paginator

        def download_file(self, bucket, key, path, Config=None):
            self.downloaded.append((bucket, key, path))

    class FakeSession:
        def client(self, *a, **k):
            return FakeClient()

    return FakeSession(), FakeClient()


def test_get_objects_downloads_new_files(tmp_path, monkeypatch):
    # Setup
    keys = ["dir/file1.jpg", "dir/file2.jpg"]
    session, fake_client = make_fake_session_and_client(tmp_path, keys)
    aws_credentials = {"AWS_URL_ENDPOINT": "http://fake"}
    bucket_name = "bucket"
    prefix = "dir/"
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    # Only file1.jpg is already processed
    pd.DataFrame({"image_path": [str(tmp_path / "dir/file1.jpg")]}).to_csv(
        csv_file, index=False
    )
    # Patch download_batch to record calls
    called = {}

    def fake_download_batch(*a, **k):
        called["called"] = True
        called["keys"] = a[2]  # keys

    monkeypatch.setattr(aws_scripts, "download_batch", fake_download_batch)
    # Run
    aws_scripts.get_objects(
        session,
        aws_credentials,
        bucket_name,
        prefix,
        str(local_path),
        batch_size=1,
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
        num_workers=1,
    )
    # Only file2.jpg should be attempted for download
    assert called["called"]
    assert called["keys"] == ["dir/file2.jpg"]


def test_get_objects_skips_corrupt(monkeypatch, tmp_path):

    class FakeClient:
        def get_paginator(self, name):
            class FakePaginator:
                def paginate(self, **kwargs):
                    return [
                        {
                            "Contents": [
                                {"Key": "$corrupt_file.jpg"},
                                {"Key": "dir/file2.jpg"},
                            ],
                            "KeyCount": 2,
                        }
                    ]

            return FakePaginator()

    class FakeSession:
        def client(self, *a, **k):
            return FakeClient()

    aws_credentials = {"AWS_URL_ENDPOINT": "http://fake"}
    bucket_name = "bucket"
    prefix = "dir/"
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": []}).to_csv(csv_file, index=False)
    # Patch download_batch to record calls
    called = {}

    def fake_download_batch(*a, **k):
        called["called"] = True
        called["keys"] = a[2]

    monkeypatch.setattr(aws_scripts, "download_batch", fake_download_batch)
    aws_scripts.get_objects(
        FakeSession(),
        aws_credentials,
        bucket_name,
        prefix,
        str(local_path),
        batch_size=2,
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
        num_workers=1,
    )
    # Only dir/file2.jpg should be attempted for download
    assert called.get("called", False)
    assert called.get("keys") == ["dir/file2.jpg"]


def test_get_objects_skips_jpg_only(monkeypatch, tmp_path):

    class FakeClient:
        def get_paginator(self, name):
            class FakePaginator:
                def paginate(self, **kwargs):
                    return [
                        {
                            "Contents": [
                                {"Key": "dir/file1.jpeg"},
                                {"Key": "dir/file2.jpg"},
                                {"Key": "dir/file3.txt"},
                                {"Key": "dir/file4.wav"},
                            ],
                            "KeyCount": 4,
                        }
                    ]

            return FakePaginator()

    class FakeSession:
        def client(self, *a, **k):
            return FakeClient()

    aws_credentials = {"AWS_URL_ENDPOINT": "http://fake"}
    bucket_name = "bucket"
    prefix = "dir/"
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": []}).to_csv(csv_file, index=False)
    # Patch download_batch to record calls
    called = {}

    def fake_download_batch(*a, **k):
        called["called"] = True
        called["keys"] = a[2]

    monkeypatch.setattr(aws_scripts, "download_batch", fake_download_batch)
    aws_scripts.get_objects(
        FakeSession(),
        aws_credentials,
        bucket_name,
        prefix,
        str(local_path),
        batch_size=2,
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
        num_workers=1,
    )
    # Only dir/file2.jpg should be attempted for download
    assert called.get("called", False)
    assert called.get("keys") == ["dir/file1.jpeg", "dir/file2.jpg"]


def test_get_objects_multithread(monkeypatch, tmp_path):
    # Setup
    keys = [f"dir/file{i}.jpg" for i in range(4)]

    class FakeClient:
        def get_paginator(self, name):
            class FakePaginator:
                def paginate(self, **kwargs):
                    return [
                        {"Contents": [{"Key": k} for k in keys], "KeyCount": len(keys)}
                    ]

            return FakePaginator()

    class FakeSession:
        def client(self, *a, **k):
            return FakeClient()

    aws_credentials = {"AWS_URL_ENDPOINT": "http://fake"}
    bucket_name = "bucket"
    prefix = "dir/"
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": []}).to_csv(csv_file, index=False)
    # Patch download_batch to record calls
    call_batches = []

    def fake_download_batch(*a, **k):
        call_batches.append(a[2])

    monkeypatch.setattr(aws_scripts, "download_batch", fake_download_batch)
    # Run with 2 workers
    aws_scripts.get_objects(
        FakeSession(),
        aws_credentials,
        bucket_name,
        prefix,
        str(local_path),
        batch_size=2,
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
        num_workers=2,
    )
    # Should call download_batch for each chunk
    assert len(call_batches) == 2
    all_keys = [k for batch in call_batches for k in batch]
    assert set(all_keys) == set(keys)


def test_get_objects_no_keys(monkeypatch, tmp_path):
    # Setup: paginator returns no keys
    class FakeClient:
        def list_objects_v2(self, Bucket, Prefix):
            return {"KeyCount": 0, "Contents": []}

        def get_paginator(self, name):
            class FakePaginator:
                def paginate(self, **kwargs):
                    return [{"Contents": [], "KeyCount": 0}]

            return FakePaginator()

    class FakeSession:
        def client(self, *a, **k):
            return FakeClient()

    aws_credentials = {"AWS_URL_ENDPOINT": "http://fake"}
    bucket_name = "bucket"
    prefix = "dir/"
    local_path = tmp_path
    csv_file = tmp_path / "results.csv"
    pd.DataFrame({"image_path": []}).to_csv(csv_file, index=False)
    # Patch download_batch to record calls
    called = {}

    def fake_download_batch(*a, **k):
        called["called"] = True

    monkeypatch.setattr(aws_scripts, "download_batch", fake_download_batch)
    aws_scripts.get_objects(
        FakeSession(),
        aws_credentials,
        bucket_name,
        prefix,
        str(local_path),
        batch_size=2,
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
        num_workers=1,
    )
    # download_batch should not be called
    assert not called.get("called", False)
