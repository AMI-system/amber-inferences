import pandas as pd
import amber_inferences.cli.get_tracks as get_tracks


def test_main_adds_tracks(tmp_path, monkeypatch, capsys):
    # Create a dummy results.csv
    df = pd.DataFrame(
        {
            "image_path": ["img1.jpg", "img2.jpg"],
            "crop_status": ["crop_1", "crop_2"],
            "x_min": [10, 20],
            "y_min": [15, 25],
            "x_max": [50, 60],
            "y_max": [55, 65],
            "other": [1, 2],
        }
    )
    csv_file = tmp_path / "results.csv"
    df.to_csv(csv_file, index=False)
    # Patch track_id_calc to return a DataFrame with track_id and crop_id
    monkeypatch.setattr(
        get_tracks,
        "track_id_calc",
        lambda df, cost_threshold=1: pd.DataFrame(
            {
                "image_path": ["img1.jpg", "img2.jpg"],
                "crop_id": ["crop_1", "crop_2"],
                "track_id": [0, 1],
            }
        ),
    )
    get_tracks.main(csv_file=csv_file, tracking_threshold=1, verbose=True)
    out = capsys.readouterr().out
    assert "Number of tracks for results.csv: 2" in out
    # Check that the output file has the new track_id column
    df_out = pd.read_csv(csv_file)
    assert "track_id" in df_out.columns
    assert set(df_out["track_id"]) == {0, 1}


def test_main_merges_correctly(tmp_path, monkeypatch):
    # Test that merge and renaming works as expected
    df = pd.DataFrame(
        {
            "image_path": ["img1.jpg"],
            "crop_status": ["crop_1"],
            "x_min": [10],
            "y_min": [15],
            "x_max": [50],
            "y_max": [55],
            "other": [1],
        }
    )
    csv_file = tmp_path / "results.csv"
    df.to_csv(csv_file, index=False)
    monkeypatch.setattr(
        get_tracks,
        "track_id_calc",
        lambda df, cost_threshold=1: pd.DataFrame(
            {
                "image_path": ["img1.jpg"],
                "crop_id": ["crop_1"],
                "x_min": [10],
                "y_min": [15],
                "x_max": [50],
                "y_max": [55],
                "track_id": [42],
            }
        ),
    )
    get_tracks.main(csv_file=csv_file, tracking_threshold=1, verbose=False)
    df_out = pd.read_csv(csv_file)
    assert "track_id" in df_out.columns
    assert df_out.loc[0, "track_id"] == 42


def test_main_handles_missing_columns(tmp_path, monkeypatch):
    # Test that main does not fail if extra columns are present
    df = pd.DataFrame(
        {
            "image_path": ["img1.jpg"],
            "crop_status": ["crop_1"],
            "other": [1],
            "x_min": [10],
            "y_min": [15],
            "x_max": [50],
            "y_max": [55],
            "image_path_basename": ["img1.jpg"],
        }
    )
    csv_file = tmp_path / "results.csv"
    df.to_csv(csv_file, index=False)
    monkeypatch.setattr(
        get_tracks,
        "track_id_calc",
        lambda df, cost_threshold=1: pd.DataFrame(
            {
                "image_path": ["img1.jpg"],
                "crop_id": ["crop_1"],
                "x_min": [10],
                "y_min": [15],
                "x_max": [50],
                "y_max": [55],
                "track_id": [7],
            }
        ),
    )
    get_tracks.main(csv_file=csv_file, tracking_threshold=1, verbose=False)
    df_out = pd.read_csv(csv_file)
    assert "track_id" in df_out.columns
    assert df_out.loc[0, "track_id"] == 7
