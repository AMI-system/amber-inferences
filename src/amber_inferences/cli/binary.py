#!/usr/bin/env python3

import argparse
import json
import os
import random
import torch
import string

from amber_inferences.utils.custom_models import load_models
from amber_inferences.utils.inference_scripts import binary_only, initialise_session


def main(
    output_dir,
    binary_model=None,
    device=None,
    crops_csv=None,
    output_csv="results.csv",
    job_name=None,
):
    """
    Main function to process a specific chunk of S3 keys.

    Args:
        chunk_id (str): ID of the chunk to process (e.g., chunk_0).
        json_file (str): Path to the JSON file with key chunks.
        output_dir (str): Directory to save results.
        bucket_name (str): S3 bucket name.
        Other args: Parameters for download and analysis.
    """

    # read in the crops csv
    crops = pd.read_csv(crops_csv)
    # drop rows where crop_status = 'NO DETECTIONS FOR IMAGE'
    crops = crops.loc[crops['crop_status'] != 'NO DETECTIONS FOR IMAGE', ]

    # for each row in the crops csv, get the image key and run the binary model
    for index, row in crops.iterrows():
        image_key = row['image_key']
        image_path = os.path.join(output_dir, image_key)
        # run the binary model
        binary, confidence, dt = binary_only(image_path, binary_model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific chunk of S3 keys.")
    parser.add_argument(
        "--crops_csv", required=True, help="Path to save analysis results."
    )
    parser.add_argument(
        "--output_csv", default="results.csv", help="Path to save analysis results."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save downloaded files and analysis results.",
        default="./data/"
    )
    parser.add_argument(
        "--job_name", default=None, help="Unique job name. If none, one will be randomly generated"
    )

    args = parser.parse_args()
    job_name = args.job_name
    if job_name is None:
        # set as a random series of alphanumeric
        job_name = ''.join(random.choice(f"{string.ascii_lowercase}{string.digits}") for i in range(16))
    args.job_name = job_name

    print(f'Saving job info to {args.output_dir}/{job_name}_job_info.json')
    with open(f"{args.output_dir}/{job_name}_job_info.json", "w") as f:
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

    # check if the model paths exist
    if not os.path.exists(os.path.abspath(args.binary_model_path)):
        raise FileNotFoundError(f"Model path not found: {args.binary_model_path}")

    if not os.path.exists(os.path.abspath(args.crops_csv)):
            raise FileNotFoundError(f"Crops csv file not found: {args.crops_csv}")


    models = load_models(
        device,
        binary_model_path=os.path.abspath(args.binary_model_path)
    )

    main(
        output_dir=args.output_dir,
        binary_model=models['binary_model'],
        device=device,
        crops_csv=args.crops_csv,
        output_csv=args.output_csv,
        job_name=args.job_name,
    )