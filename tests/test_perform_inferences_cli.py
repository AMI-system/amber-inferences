import subprocess
import sys
import json
from pathlib import Path

CLI_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "amber_inferences"
    / "cli"
    / "perform_inferences.py"
)


def make_dummy_json(tmp_path):
    # Create a dummy chunked json file
    data = {"2021-01-01": ["img1.jpg", "img2.jpg"]}
    json_path = tmp_path / "chunks.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def make_dummy_model(tmp_path):
    # Create dummy model files with correct extensions
    files = {}
    for name in [
        "localisation_model_path",
        "binary_model_path",
        "order_model_path",
        "species_model_path",
    ]:
        p = tmp_path / f"{name}.pt"
        p.write_text("dummy")
        files[name] = p
    # species_labels should be a .json file
    labels_path = tmp_path / "species_labels.json"
    labels_path.write_text("{}")
    order_thresholds_path = tmp_path / "order_thresholds.csv"
    order_thresholds_path.write_text("{}")
    files["order_thresholds_path"] = order_thresholds_path
    files["species_labels"] = labels_path
    return files


def test_cli_main_run_fails(tmp_path):
    json_file = make_dummy_json(tmp_path)
    model_files = make_dummy_model(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv_file = tmp_path / "results.csv"
    # Run the CLI with minimal required args
    cmd = [
        sys.executable,
        str(CLI_PATH),
        "--chunk_id",
        "1",
        "--json_file",
        str(json_file),
        "--output_dir",
        str(output_dir),
        "--bucket_name",
        "bucket",
        "--localisation_model_path",
        str(model_files["localisation_model_path"]),
        "--binary_model_path",
        str(model_files["binary_model_path"]),
        "--order_model_path",
        str(model_files["order_model_path"]),
        "--order_thresholds_path",
        "wrong_extension.txt",  # Invalid extension
        "--species_model_path",
        str(model_files["species_model_path"]),
        "--species_labels",
        str(model_files["species_labels"]),
        "--csv_file",
        str(csv_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert "File not found" in result.stderr


def test_cli_main_run(tmp_path):
    json_file = make_dummy_json(tmp_path)
    model_files = make_dummy_model(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv_file = tmp_path / "results.csv"
    # Run the CLI with minimal required args
    cmd = [
        sys.executable,
        str(CLI_PATH),
        "--chunk_id",
        "1",
        "--json_file",
        str(json_file),
        "--output_dir",
        str(output_dir),
        "--bucket_name",
        "bucket",
        "--localisation_model_path",
        str(model_files["localisation_model_path"]),
        "--binary_model_path",
        str(model_files["binary_model_path"]),
        "--order_model_path",
        str(model_files["order_model_path"]),
        "--order_thresholds_path",
        str(model_files["order_thresholds_path"]),
        "--species_model_path",
        str(model_files["species_model_path"]),
        "--species_labels",
        str(model_files["species_labels"]),
        "--csv_file",
        str(csv_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0  # 0 = success, shouldnt work since dummy models


def test_cli_argparse_errors(tmp_path):
    # Missing required args should fail
    cmd = [sys.executable, str(CLI_PATH)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "usage:" in result.stderr or "usage:" in result.stdout
