import pytest
from unittest import mock

import amber_inferences.utils.deployment_summary as deployment_summary


def make_fake_deployments():
    return [
        {
            "deployment_id": "dep1",
            "status": "active",
            "country": "UK",
            "country_code": "UK",
            "location_name": "loc1",
        },
        {
            "deployment_id": "dep2",
            "status": "inactive",
            "country": "UK",
            "country_code": "UK",
            "location_name": "loc2",
        },
        {
            "deployment_id": "dep3",
            "status": "active",
            "country": "SG",
            "country_code": "SG",
            "location_name": "loc3",
        },
    ]


def test_get_deployments_success(monkeypatch):
    monkeypatch.setattr(
        deployment_summary.requests,
        "post",
        lambda *a, **k: type(
            "Resp",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: make_fake_deployments(),
            },
        )(),
    )
    out = deployment_summary.get_deployments("user", "pass")
    assert isinstance(out, list)
    assert any(d["deployment_id"] == "dep1" for d in out)


def test_get_deployments_http_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise deployment_summary.requests.exceptions.HTTPError("401 error")

    monkeypatch.setattr(
        deployment_summary.requests, "post", lambda *a, **k: FakeResponse()
    )
    with pytest.raises(SystemExit):
        deployment_summary.get_deployments("user", "pass")


def test_count_files(monkeypatch):
    s3_client = mock.Mock()
    s3_client.list_objects_v2.side_effect = [
        {"Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}], "IsTruncated": False}
    ]
    result = deployment_summary.count_files(s3_client, "bucket", "prefix")
    assert result["image_count"] == 1
    assert result["audio_count"] == 1
    assert "a.jpg" in result["keys"]


def test_deployment_data(monkeypatch):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(creds, include_inactive=False)
    assert "dep1" in out
    assert out["dep1"]["image_count"] == 1
    assert out["dep1"]["audio_count"] == 1


def test_deployment_data_subset_deployment_id(monkeypatch):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_deployments=["dep1"], include_file_count=True
    )
    assert list(out.keys()) == ["dep1"]
    assert out["dep1"]["image_count"] == 1
    assert out["dep1"]["audio_count"] == 1


def test_invalid_deployment_subset(monkeypatch):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_deployments=["dep_none"], include_file_count=True
    )
    assert out == {}  # Should return empty dict if no matching deployment_id


def test_deployment_data_subset_country(monkeypatch):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_countries=["SG"], include_file_count=True
    )
    assert list(out.keys()) == ["dep3"]
    assert out["dep3"]["country"] == "Sg"
    assert out["dep3"]["image_count"] == 1
    assert out["dep3"]["audio_count"] == 1


def test_deployment_data_invalid_subset_country(monkeypatch):
    """
    Test that deployment_data returns empty dict for invalid subset_countries.
    """
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_countries=["Atlantis"], include_file_count=True
    )
    assert out == {}


def test_deployment_data_inactive(monkeypatch):
    """
    Test that deployment_data includes inactive deployments when specified.
    """
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, include_inactive=True, include_file_count=True
    )
    assert set(out.keys()) == {"dep1", "dep2", "dep3"}
    assert out["dep2"]["status"] == "inactive"

    out2 = deployment_summary.deployment_data(
        creds, include_inactive=False, include_file_count=True
    )
    assert set(out2.keys()) == {"dep1", "dep3"}


def test_print_deployments(monkeypatch, capsys):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary, "get_deployments", lambda u, p: make_fake_deployments()
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {
                        "Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}],
                        "IsTruncated": False,
                    }

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    deployment_summary.print_deployments(
        creds, include_inactive=False, subset_countries=None, print_file_count=True
    )
    out = capsys.readouterr().out
    assert "Deployment ID: dep1" in out
    assert "images and" in out
