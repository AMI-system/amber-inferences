import pytest
from unittest import mock

import amber_inferences.utils.api_utils as api_utils


def test_get_buckets():
    """Test that get_buckets returns a list of bucket names."""
    s3_client = mock.Mock()
    s3_client.list_buckets.return_value = {"Buckets": [{"Name": "b1"}, {"Name": "b2"}]}
    buckets = api_utils.get_buckets(s3_client)
    assert buckets == ["b1", "b2"]


def test_get_deployments_success(monkeypatch):
    """Test that get_deployments returns a list of deployments."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"deployment_id": "d1"}]

    monkeypatch.setattr(api_utils.requests, "post", lambda *a, **k: FakeResponse())
    result = api_utils.get_deployments("user", "pass")
    assert result == [{"deployment_id": "d1"}]


def test_get_deployment_names_success(monkeypatch):
    """Test that get_deployments returns a list of deployments."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"country_code": "gbr", "deployment_id": "dep000001"},
                {"country_code": "gbr", "deployment_id": "dep000002"},
                {"country_code": "cri", "deployment_id": "dep000031"},
                {"country_code": "cri", "deployment_id": "dep000032"},
                {"country_code": "sgp", "deployment_id": "dep000050"},
            ]

    monkeypatch.setattr(api_utils.requests, "post", lambda *a, **k: FakeResponse())
    result = api_utils.get_deployment_names("user", "pass", "cri")
    assert "dep000031" in result
    assert "dep000001" not in result
    assert "dep000050" not in result
    assert "dep000032" in result


def test_get_deployment_names_no_bucket(monkeypatch, capsys):
    """Test that get_deployment_names prints an error if bucket not in object store."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"country_code": "gbr", "deployment_id": "dep000001"},
                {"country_code": "gbr", "deployment_id": "dep000002"},
                {"country_code": "cri", "deployment_id": "dep000031"},
                {"country_code": "cri", "deployment_id": "dep000032"},
                {"country_code": "sgp", "deployment_id": "dep000050"},
            ]

    monkeypatch.setattr(api_utils.requests, "post", lambda *a, **k: FakeResponse())
    result = api_utils.get_deployment_names("user", "pass", "atlanta")
    assert "No deployments found for bucket:" in capsys.readouterr().out
    assert result == []


def test_get_deployments_http_error(monkeypatch, capsys):
    """Test that get_deployments handles HTTP errors."""

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise api_utils.requests.exceptions.HTTPError("401 error")

    def fake_post(*a, **k):
        return FakeResponse()

    monkeypatch.setattr(api_utils.requests, "post", fake_post)
    with pytest.raises(SystemExit):
        api_utils.get_deployments("user", "pass")
    out = capsys.readouterr().out
    assert "HTTP Error" in out or "Wrong username or password" in out


def test_get_deployments_other_error(monkeypatch, capsys):
    """Test that get_deployments handles other (non HTTP) errors."""

    def fake_post(*a, **k):
        raise Exception("fail")

    monkeypatch.setattr(api_utils.requests, "post", fake_post)
    with pytest.raises(SystemExit):
        api_utils.get_deployments("user", "pass")
    out = capsys.readouterr().out
    assert "Error:" in out


def test_count_files():
    """Test that count_files counts files by type correctly."""
    s3_client = mock.Mock()
    paginator = mock.Mock()
    s3_client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "a.jpg"},
                {"Key": "b.wav"},
                {"Key": "c.txt"},
                {"Key": "d.jpg"},
            ]
        }
    ]
    result = api_utils.count_files(s3_client, "bucket", "prefix")
    assert result["image_count"] == 2
    assert result["audio_count"] == 1
    assert result["other_count"] == 1
    assert "txt" in result["other_file_types"]
