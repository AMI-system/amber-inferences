#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import os

from amber_inferences.utils.tracking import track_id_calc


def main(csv_file=Path("results.csv"), tracking_threshold=1, verbose=True):
    """
    Main function to add tracks to the results CSV file.
    """

    csv_file = Path(csv_file)
    df = pd.read_csv(csv_file)

    # remove rows that were rerun to avoid tracking discontinuity
    df = df.drop_duplicates(
        subset=["image_path", "crop_status", "x_min", "y_min", "x_max", "y_max"],
        keep="first",
    )

    track_df = track_id_calc(df, cost_threshold=tracking_threshold)

    if verbose:
        print(
            f"Number of tracks for {os.path.basename(csv_file)}: {track_df['track_id'].nunique()}"
        )

    df["image_path_basename"] = df["image_path"].apply(lambda x: os.path.basename(x))

    df = df.merge(
        track_df,
        how="left",
        left_on=["image_path_basename", "crop_status"],
        right_on=["image_path", "crop_id"],
    )
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("_y")]
    df = df.rename(columns=lambda x: x.replace("_x", ""))

    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the object tracks.")
    parser.add_argument(
        "--tracking_threshold",
        type=float,
        default=1,
        help="Threshold for the track cost. Default: 1",
    )
    parser.add_argument(
        "--csv_file",
        default=Path("results.csv"),
        help="Path to save analysis results.",
        type=Path,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose statements.",
    )

    args = parser.parse_args()

    main(
        csv_file=args.csv_file,
        tracking_threshold=args.tracking_threshold,
        verbose=args.verbose,
    )
