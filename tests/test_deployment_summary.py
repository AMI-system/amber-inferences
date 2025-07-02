import pytest
from unittest import mock

import amber_inferences.utils.deployment_summary as deployment_summary
from amber_inferences.utils.api_utils import get_deployments


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
    out = get_deployments("user", "pass")
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
        get_deployments("user", "pass")


def test_count_files(monkeypatch):
    s3_client = mock.Mock()
    s3_client.list_objects_v2.side_effect = [
        {"Contents": [{"Key": "a.jpg"}, {"Key": "b.wav"}], "IsTruncated": False}
    ]
    result = deployment_summary.count_files(s3_client, "bucket", "prefix")
    assert result["image_count"] == 1
    assert result["audio_count"] == 1
    assert "a.jpg" in result["keys"]


def test_count_files_empty(monkeypatch):
    s3_client = mock.Mock()
    s3_client.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    result = deployment_summary.count_files(s3_client, "bucket", "prefix")
    assert result["image_count"] == 0
    assert result["audio_count"] == 0
    assert result["keys"] == []


def test_count_files_multiple_pages(monkeypatch):
    s3_client = mock.Mock()
    s3_client.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "a.jpg"}],
            "IsTruncated": True,
            "NextContinuationToken": "tok1",
        },
        {"Contents": [{"Key": "b.wav"}], "IsTruncated": False},
    ]
    result = deployment_summary.count_files(s3_client, "bucket", "prefix")
    assert set(result["keys"]) == {"a.jpg", "b.wav"}
    assert result["image_count"] == 1
    assert result["audio_count"] == 1


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


def test_deployment_data_country_code(monkeypatch):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "dep1",
                "status": "active",
                "country": "UK",
                "country_code": "GB",
                "location_name": "loc1",
            },
            {
                "deployment_id": "dep2",
                "status": "active",
                "country": "Panama",
                "country_code": "PA",
                "location_name": "loc2",
            },
        ],
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {"Contents": [], "IsTruncated": False}

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_countries=["GB"], include_file_count=False
    )
    assert "dep1" in out
    assert out["dep1"]["country_code"] == "GB"


def test_deployment_data_subset_deployments_empty(monkeypatch, capsys):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "dep1",
                "status": "active",
                "country": "UK",
                "country_code": "UK",
                "location_name": "loc1",
            }
        ],
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {"Contents": [], "IsTruncated": False}

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    out = deployment_summary.deployment_data(
        creds, subset_deployments=[], include_file_count=False
    )
    assert out == {}
    assert "No deployments found" in capsys.readouterr().out


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


def test_print_deployments_country_warning(monkeypatch, capsys):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "dep1",
                "status": "active",
                "country": "UK",
                "country_code": "UK",
                "location_name": "loc1",
            }
        ],
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {"Contents": [], "IsTruncated": False}

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    deployment_summary.print_deployments(creds, subset_countries=["Atlantis"])
    out = capsys.readouterr().out
    assert "WARNING: Atlantis does not have any" in out


def test_print_deployments_actuve(monkeypatch, capsys):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "dep1",
                "status": "inactive",
                "country": "UK",
                "country_code": "UK",
                "location_name": "loc1",
            },
            {
                "deployment_id": "dep2",
                "status": "active",
                "country": "UK",
                "country_code": "UK",
                "location_name": "loc2",
            },
        ],
    )

    class FakeSession:
        def client(self, *a, **k):
            class FakeS3:
                def list_objects_v2(self, **kwargs):
                    return {"Contents": [], "IsTruncated": False}

            return FakeS3()

    monkeypatch.setattr(
        deployment_summary.boto3, "Session", lambda **kwargs: FakeSession()
    )
    deployment_summary.print_deployments(
        creds, subset_countries=["UK"], include_inactive=True
    )
    out = capsys.readouterr().out
    assert "active deployments" not in out
    assert "2 deployments" in out


def test_print_deployments_print_file_count(monkeypatch, capsys):
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(
        deployment_summary,
        "get_deployments",
        lambda u, p: [
            {
                "deployment_id": "dep1",
                "status": "active",
                "country": "UK",
                "country_code": "UK",
                "location_name": "loc1",
            }
        ],
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
    deployment_summary.print_deployments(creds, print_file_count=True)
    out = capsys.readouterr().out
    assert " - This deployment has 1 images and 1 audio files." in out
    assert "Uk (UK) has 1 active deployment" in out


def test_main(monkeypatch, tmp_path):
    # Patch print_deployments to record call
    called = {}

    def fake_print_deployments(*a, **k):
        called["called"] = True
        called["args"] = a
        called["kwargs"] = k

    monkeypatch.setattr(deployment_summary, "print_deployments", fake_print_deployments)
    creds = {
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    cred_path = tmp_path / "creds.json"
    import json

    with open(cred_path, "w") as f:
        json.dump(creds, f)
    import sys

    sys_argv = sys.argv
    sys.argv = ["prog", "--credentials_file", str(cred_path)]
    try:
        deployment_summary.main()
    finally:
        sys.argv = sys_argv
    assert called["called"]
