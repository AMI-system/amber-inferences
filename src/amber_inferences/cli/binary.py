#!/usr/bin/env python3

import argparse
import json
import random
import torch
import string
from pathlib import Path
import pandas as pd

from amber_inferences.utils.custom_models import load_models
from amber_inferences.utils.inference_scripts import classify_box


def main(
    output_dir,
    binary_model=None,
    crops_csv=None,
    output_csv=Path("results.csv"),
):
    """
    Main function to process a specific chunk of S3 keys.
    """
    output_dir = Path(output_dir)
    crops_csv = Path(crops_csv)
    output_csv = Path(output_csv)

    # read in the crops csv
    crops = pd.read_csv(crops_csv)
    # drop rows where crop_status = 'NO DETECTIONS FOR IMAGE'
    crops = crops.loc[crops["crop_status"] != "No detections for image.",]

    results = []
    for _, row in crops.iterrows():
        image_key = row["image_key"]
        image_path = output_dir / image_key
        # run the binary model
        # classify_box should return (label, confidence)
        label, confidence = classify_box(image_path, binary_model)
        results.append(
            {"image_key": image_key, "label": label, "confidence": confidence}
        )
    # Save results to output_csv
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific chunk of S3 keys.")
    parser.add_argument(
        "--crops_csv", required=True, help="Path to crops CSV file.", type=Path
    )
    parser.add_argument(
        "--output_csv",
        default=Path("results.csv"),
        help="Path to save analysis results.",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save downloaded files and analysis results.",
        default=Path("./data/"),
        type=Path,
    )
    parser.add_argument(
        "--binary_model_path",
        required=True,
        type=Path,
        help="Path to the binary model weights.",
    )
    parser.add_argument(
        "--job_name",
        default=None,
        help="Unique job name. If none, one will be randomly generated",
    )

    args = parser.parse_args()
    job_name = args.job_name
    if job_name is None:
        job_name = "".join(
            random.choice(f"{string.ascii_lowercase}{string.digits}") for _ in range(16)
        )
    args.job_name = job_name

    print(f"Saving job info to {args.output_dir}/{job_name}_job_info.json")
    with open(args.output_dir / f"{job_name}_job_info.json", "w") as f:
        json.dump(vars(args), f)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(
            "\033[95m\033[1mCuda available, using GPU "
            + "\N{White Heavy Check Mark}\033[0m\033[0m"
        )
    else:
        device = torch.device("cpu")
        print(
            "\033[95m\033[1mCuda not available, using CPU "
            + "\N{Cross Mark}\033[0m\033[0m"
        )

    if not args.binary_model_path.resolve().exists():
        raise FileNotFoundError(f"Model path not found: {args.binary_model_path}")
    if not args.crops_csv.resolve().exists():
        raise FileNotFoundError(f"Crops csv file not found: {args.crops_csv}")

    models = load_models(device, binary_model_path=args.binary_model_path.resolve())

    main(
        output_dir=args.output_dir,
        binary_model=models["classification_model"],
        crops_csv=args.crops_csv,
        output_csv=args.output_csv,
        job_name=args.job_name,
    )
