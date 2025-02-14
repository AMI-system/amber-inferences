#!/usr/bin/env python3

import argparse
import json
import os
import random
import boto3
import torch

from amber_inferences.utils.custom_models import load_models
from amber_inferences.utils.inference_scripts import localisation_only, initialise_session

def main(
    chunk_id,
    json_file,
    output_dir,
    bucket_name,
    credentials_file="credentials.json",
    remove_image=True,
    perform_inference=True,
    save_crops=False,
    localisation_model=None,
    box_threshold=0.99,
    device=None,
    csv_file="results.csv",
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
    with open(json_file, "r") as f:
        chunks = json.load(f)

    if chunk_id not in chunks:
        raise ValueError(f"Chunk ID {chunk_id} not found in JSON file.")

    client = initialise_session(credentials_file)

    keys = chunks[chunk_id]["keys"]
    localisation_only(
        keys=keys,
        output_dir=output_dir,
        bucket_name=bucket_name,
        client=client,
        remove_image=remove_image,
        perform_inference=perform_inference,
        save_crops=save_crops,
        localisation_model=localisation_model,
        box_threshold=box_threshold,
        device=device,
        csv_file=csv_file,job_name=job_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific chunk of S3 keys.")
    parser.add_argument(
        "--chunk_id",
        required=True,
        help="ID of the chunk to process (e.g., 0, 1, 2, 3).",
    )
    parser.add_argument(
        "--json_file", required=True, help="Path to the JSON file with key chunks."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save downloaded files and analysis results.",
        default="./data/",
    )
    parser.add_argument("--bucket_name", required=True, help="Name of the S3 bucket.")
    parser.add_argument(
        "--credentials_file",
        default="credentials.json",
        help="Path to AWS credentials file.",
    )
    parser.add_argument(
        "--remove_image", action="store_true", help="Remove images after processing."
    )
    parser.add_argument(
        "--perform_inference", action="store_true", help="Enable inference."
    )
    parser.add_argument(
        "--save_crops", action="store_true", help="Whether to save the crops."
    )
    # TODO: ensure pipeline works with flatbug
    parser.add_argument(
        "--localisation_model_path",
        type=str,
        required=True,
        help="Path to the localisation model weights.",
        default=None,
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.99,
        help="Threshold for the confidence score of bounding boxes. Default: 0.99",
    )
    parser.add_argument(
        "--csv_file", default="results.csv", help="Path to save analysis results."
    )
    parser.add_argument(
        "--job_name", default=None, help="Unique job name. If none, one will be randomly generated"
    )

    args = parser.parse_args()
    job_name = args.job_name
    if job_name is None:
        # set as a random series of alphanumeric
        job_name = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
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
    if not os.path.exists(os.path.abspath(args.localisation_model_path)):
        raise FileNotFoundError(f"Model path not found: {args.localisation_model_path}")

    if not os.path.exists(os.path.abspath(args.json_file)):
            raise FileNotFoundError(f"JSON file not found: {args.json_file}")


    models = load_models(
        device,
        localisation_model_path=os.path.abspath(args.localisation_model_path)
    )

    main(
        chunk_id=args.chunk_id,
        json_file=args.json_file,
        output_dir=args.output_dir,
        bucket_name=args.bucket_name,
        credentials_file=args.credentials_file,
        remove_image=args.remove_image,
        save_crops=args.save_crops,
        perform_inference=args.perform_inference,
        localisation_model=models["localisation_model"],
        box_threshold=args.box_threshold,
        device=device,
        csv_file=args.csv_file,
        job_name=job_name
    )
