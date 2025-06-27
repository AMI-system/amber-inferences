import types
import amber_inferences.cli.deployments as deployments


def test_main_calls_deployment_data(monkeypatch, capsys):
    args = types.SimpleNamespace(
        include_inactive=True,
        print_file_count=True,
        subset_countries=["UK", "SG"],
        subset_deployments=["dep1"],
    )
    monkeypatch.setattr(
        deployments.argparse.ArgumentParser, "parse_args", lambda self: args
    )
    monkeypatch.setattr(deployments, "load_credentials", lambda: {"fake": "creds"})
    fake_summary = {
        "dep1": {"deployment_id": "dep1", "location_name": "UK", "n_images": 42},
        "dep2": {"deployment_id": "dep2", "location_name": "SG"},
    }
    monkeypatch.setattr(
        deployments.deployment_summary, "deployment_data", lambda *a, **k: fake_summary
    )
    deployments.main()
    out = capsys.readouterr().out
    assert "Deployment ID: dep1 - Location: UK" in out
    assert "This deployment has 42 images" in out
    assert "Deployment ID: dep2 - Location: SG" in out


def test_main_no_images(monkeypatch, capsys):
    args = types.SimpleNamespace(
        include_inactive=False,
        print_file_count=False,
        subset_countries=None,
        subset_deployments=None,
    )
    monkeypatch.setattr(
        deployments.argparse.ArgumentParser, "parse_args", lambda self: args
    )
    monkeypatch.setattr(deployments, "load_credentials", lambda: {"fake": "creds"})
    fake_summary = {
        "dep1": {"deployment_id": "dep1", "location_name": "UK"},
    }
    monkeypatch.setattr(
        deployments.deployment_summary, "deployment_data", lambda *a, **k: fake_summary
    )
    deployments.main()
    out = capsys.readouterr().out
    assert "Deployment ID: dep1 - Location: UK" in out
    assert "images" not in out
