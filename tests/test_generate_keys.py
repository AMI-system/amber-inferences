from unittest import mock
import types

import amber_inferences.cli.generate_keys as generate_keys


def test_main_calls_save_keys(monkeypatch):
    "Test that the main function calls save_keys with the correct parameters."
    # Mock the argparse
    args = types.SimpleNamespace(
        bucket="test-bucket",
        deployment_id="dep123",
        output_file="out.txt",
        file_extensions=["jpg", "jpeg"],
    )
    monkeypatch.setattr(
        generate_keys.argparse.ArgumentParser, "parse_args", lambda self: args
    )

    # Mock credentials and boto3
    fake_creds = {
        "AWS_ACCESS_KEY_ID": "id",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "region",
        "AWS_URL_ENDPOINT": "endpoint",
    }
    monkeypatch.setattr(generate_keys, "load_credentials", lambda: fake_creds)
    fake_session = mock.Mock()
    fake_client = mock.Mock()
    fake_session.client.return_value = fake_client
    monkeypatch.setattr(
        generate_keys, "boto3", mock.Mock(Session=lambda **kwargs: fake_session)
    )

    called = {}

    def fake_save_keys(s3_client, bucket, deployment_id, output_file, verbose):
        called.update(locals())

    monkeypatch.setattr(generate_keys, "save_keys", fake_save_keys)

    generate_keys.main()
    assert called["bucket"] == "test-bucket"
    assert called["deployment_id"] == "dep123"
    assert called["output_file"] == "out.txt"
    assert called["verbose"] is True
    assert called["s3_client"] == fake_client


def test_main_prints(monkeypatch, capsys):
    "Test that the main function prints the expected output."
    args = types.SimpleNamespace(
        bucket="bucket",
        deployment_id="dep",
        output_file="out.txt",
        file_extensions=["jpg"],
    )
    monkeypatch.setattr(
        generate_keys.argparse.ArgumentParser, "parse_args", lambda self: args
    )
    monkeypatch.setattr(
        generate_keys,
        "load_credentials",
        lambda: {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": "region",
            "AWS_URL_ENDPOINT": "endpoint",
        },
    )
    fake_session = mock.Mock()
    fake_client = mock.Mock()
    fake_session.client.return_value = fake_client
    monkeypatch.setattr(
        generate_keys, "boto3", mock.Mock(Session=lambda **kwargs: fake_session)
    )
    monkeypatch.setattr(generate_keys, "save_keys", lambda *a, **k: None)
    generate_keys.main()
    out = capsys.readouterr().out
    assert "Listing keys from bucket" in out
