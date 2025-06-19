import pytest
import json

import amber_inferences.utils.config as config


def test_validate_aws_credentials_valid():
    creds = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "url",
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
    }
    config.validate_aws_credentials(creds)  # Should not raise


def test_validate_aws_credentials_missing():
    creds = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "url",
        "UKCEH_username": "user",
    }  # missing password
    with pytest.raises(ValueError) as e:
        config.validate_aws_credentials(creds)
    assert "Missing required credentials" in str(e.value)


def test_validate_aws_credentials_empty():
    creds = {
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "url",
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
    }
    with pytest.raises(ValueError) as e:
        config.validate_aws_credentials(creds)
    assert "must be a non-empty string" in str(e.value)


def test_load_credentials_success(tmp_path):
    creds = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "url",
        "UKCEH_username": "user",
        "UKCEH_password": "pass",
    }
    cred_path = tmp_path / "creds.json"
    with open(cred_path, "w", encoding="utf-8") as f:
        json.dump(creds, f)
    loaded = config.load_credentials(str(cred_path))
    assert loaded == creds


def test_load_credentials_missing_file(tmp_path):
    cred_path = tmp_path / "doesnotexist.json"
    with pytest.raises(FileNotFoundError):
        config.load_credentials(str(cred_path))
