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


def test_deployments_summary(monkeypatch):
    """Test that deployments_summary returns a summary of deployments."""
    aws_credentials = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
    }
    monkeypatch.setattr(
        api_utils,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "d1",
                "status": "active",
                "country": "uk",
                "country_code": "UK",
            }
        ],
    )
    monkeypatch.setattr(
        api_utils,
        "count_files",
        lambda *a, **k: {
            "image_count": 1,
            "audio_count": 0,
            "other_count": 0,
            "other_file_types": [],
        },
    )

    class FakeSession:
        def client(self, *a, **k):
            return mock.Mock()

    monkeypatch.setattr(api_utils.boto3, "Session", lambda **kwargs: FakeSession())
    result = api_utils.deployments_summary(aws_credentials)
    assert "d1" in result
    assert result["d1"]["file_types"]["image_count"] == 1
