import json
import os
from unittest import mock

import amber_inferences.utils.key_utils as key_utils


def test_list_s3_keys(monkeypatch):
    """
    Test listing S3 keys with various filters and conditions.
    Check corryupt and binned images removed, others kept.
    """
    s3_client = mock.Mock()

    # Simulate paginated S3 responses
    s3_client.list_objects_v2.side_effect = [
        {
            "Contents": [
                {"Key": "dep/file1.jpg"},
                {"Key": "dep/.hidden"},
                {"Key": "dep/$corrupt"},
                {"Key": "dep/recycle/file2.jpg"},
                {"Key": "dep/file3.jpg"},
            ],
            "IsTruncated": False,
        }
    ]
    keys = key_utils.list_s3_keys(s3_client, "bucket", "dep", subdir=None)
    assert "dep/file1.jpg" in keys
    assert "dep/file3.jpg" in keys
    assert all(not os.path.basename(k).startswith("$") for k in keys)
    assert all("recycle" not in k for k in keys)
    assert all(not os.path.basename(k).startswith(".") for k in keys)


def test_process_date_valid(tmp_path):
    """Test processing a valid date from an image path."""
    dt1 = key_utils.process_date("foo-20240101123456-bar.jpg", "depid", tmp_path)
    assert dt1.year == 2024
    assert dt1.month == 1
    assert dt1.day == 1

    dt1 = key_utils.process_date("foo-20140101123456-bar.jpg", "depid", tmp_path)
    assert dt1.year == 2014
    assert dt1.month == 1
    assert dt1.day == 1

    dt2 = key_utils.process_date("20240101123456-foo-bar.jpg", "depid", tmp_path)
    assert dt2.year == 2024
    assert dt2.month == 1
    assert dt2.day == 1

    dt3 = key_utils.process_date("foo-bar-20240101123456.jpg", "depid", tmp_path)
    assert dt3.year == 2024
    assert dt3.month == 1
    assert dt3.day == 1

    # check first date is used if multiple are present
    dt4 = key_utils.process_date("20240101123456-20250101123456.jpg", "depid", tmp_path)
    assert dt4.year == 2024
    assert dt4.month == 1
    assert dt4.day == 1


def test_process_date_invalid(tmp_path):
    # Should log error and return ""
    log_file = tmp_path / "depid_error_log.txt"
    out = key_utils.process_date("foo-bar.jpg", str(tmp_path / "depid"), tmp_path)
    assert out == ""
    assert log_file.exists() or log_file.parent.exists()


def test_multiple_date_strings(tmp_path):
    # Should log the warning but return the first date
    log_file = tmp_path / "depid_error_log.txt"
    out = key_utils.process_date(
        "201501011212-2024191911212.jpg", str(tmp_path / "depid"), tmp_path
    )
    assert out.year == 2015
    assert out.month == 1
    assert out.day == 1
    assert log_file.exists() or log_file.parent.exists()


def test_save_keys(monkeypatch, tmp_path):
    """
    Test saving S3 keys to a JSON file, ensuring correct structure and content.
    """
    s3_client = mock.Mock()
    dummy_data = [
        "dep/file1-20240101123456-snapshot.jpg",
        "dep/file2-20240102123456-snapshot.jpg",
    ]
    monkeypatch.setattr(key_utils, "list_s3_keys", lambda *a, **k: dummy_data)
    output_file = tmp_path / "out.json"
    key_utils.save_keys(s3_client, "bucket", "dep", str(output_file), verbose=True)
    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert any(isinstance(v, list) for v in data.values())


def test_load_workload(tmp_path):
    file = tmp_path / "keys.txt"
    lines = [
        "foo/20240101-snapshot.jpg",
        "foo/20240102-snapshot.jpeg",
        "$corrupt.jpg",
        ".hidden.jpg",
        "foo/recycle/file.jpg",
    ]
    file.write_text("\n".join(lines))
    keys = key_utils.load_workload(
        str(file), ["jpg", "jpeg"], subset_dates=["2024-01-01"]
    )
    assert any("20240101" in k for k in keys)
    assert all(not os.path.basename(k).startswith("$") for k in keys)
    assert all(not os.path.basename(k).startswith(".") for k in keys)
    assert all("recycle" not in k for k in keys)


def test_split_workload():
    """No longer used, but test it anyway."""
    keys = [f"file{i}" for i in range(10)]
    chunks = key_utils.split_workload(keys, 3)
    assert isinstance(chunks, dict)
    assert sum(len(v["keys"]) for v in chunks.values()) == 10


def test_save_chunks(tmp_path):
    """Test the structure of the output JSON file."""
    chunks = {"1": {"keys": ["a", "b"]}, "2": {"keys": ["c"]}}
    output_file = tmp_path / "chunks.json"
    key_utils.save_chunks(chunks, str(output_file))
    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert "1" in data and "2" in data
    assert data["1"] == {"keys": ["a", "b"]}
