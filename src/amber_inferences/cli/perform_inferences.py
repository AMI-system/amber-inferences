#!/usr/bin/env python3

import argparse
import json
import os
import torch
import pandas as pd

from amber_inferences.utils.custom_models import load_models
from amber_inferences.utils.inference_scripts import (
    download_and_analyse,
    initialise_session,
)


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
    binary_model=None,
    order_model=None,
    order_labels=None,
    species_model=None,
    species_labels=None,
    device=None,
    order_data_thresholds=None,
    top_n=5,
    csv_file="results.csv",
    skip_processed=False,
    verbose=False,
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
    session_dates = list(chunks.keys())

    client = initialise_session(credentials_file)

    try:
        chunk_id = int(chunk_id) - 1  # Convert to zero-based index
        keys = chunks[session_dates[chunk_id]]
    except ValueError as e:
        raise ValueError(
            f"{e}: Chunk ID {chunk_id} was not indexable in {json_file} (json len={len(chunks)})."
        )

    # if the csv files exists, and skip_processed is set to true, then remove keys which are already in the csv
    if os.path.exists(csv_file) and skip_processed:
        already_processed = pd.read_csv(csv_file)
        csv_keys = already_processed["image_path"].tolist()
        csv_keys = [os.path.basename(key) for key in csv_keys]
        keys = [key for key in keys if os.path.basename(key) not in csv_keys]

        if len(csv_keys) > 0 and verbose:
            print(
                f"\033[93m\033[1mSkipping {len(csv_keys)} images previously processed. "
                + "\033[0m\033[0m"
            )

    # exit if length keys is 0
    if len(keys) == 0:
        print(
            f"\033[93m\033[1mAll images already processed in {csv_file}"
            + "\N{Warning Sign}\033[0m\033[0m"
        )
        return

    download_and_analyse(
        keys=keys,
        output_dir=output_dir,
        bucket_name=bucket_name,
        client=client,
        remove_image=remove_image,
        perform_inference=perform_inference,
        save_crops=save_crops,
        localisation_model=localisation_model,
        box_threshold=box_threshold,
        binary_model=binary_model,
        order_model=order_model,
        order_labels=order_labels,
        species_model=species_model,
        species_labels=species_labels,
        device=device,
        order_data_thresholds=order_data_thresholds,
        top_n=top_n,
        csv_file=csv_file,
        verbose=verbose,
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
        "--binary_model_path",
        type=str,
        help="Path to the binary model weights.",
        default="./models/moth-nonmoth-effv2b3_20220506_061527_30.pth",
    )
    parser.add_argument(
        "--order_model_path",
        type=str,
        help="Path to the order model weights.",
        default="./models/dhc_best_128.pth",
    )
    parser.add_argument(
        "--order_labels", type=str, help="Path to the order labels file."
    )
    parser.add_argument(
        "--species_model_path",
        type=str,
        help="Path to the species model weights.",
        default="./models/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt",
    )
    parser.add_argument(
        "--species_labels",
        type=str,
        help="Path to the species labels file.",
        default="./models/03_costarica_data_category_map.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--order_thresholds_path",
        type=str,
        help="Path to the order data thresholds file.",
        default="./models/thresholdsTestTrain.csv",
    )
    parser.add_argument(
        "--top_n_species",
        type=int,
        help="The number of predictions to output.",
        default=5,
    )
    parser.add_argument(
        "--csv_file", default="results.csv", help="Path to save analysis results."
    )
    parser.add_argument(
        "--skip_processed",
        action="store_true",
        help="Whether to rerun inferences for images which have already been processed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose statements.",
    )

    args = parser.parse_args()

    # if skip_processed is false, print that those will be skipped
    if args.skip_processed:
        print(
            "\n\033[95m\033[1mNote: Images already processed will be skipped. "
            + "\033[0m\033[0m"
        )

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
    for mod_path in [
        args.localisation_model_path,
        args.binary_model_path,
        args.order_model_path,
        args.order_thresholds_path,
        args.species_model_path,
        args.species_labels,
    ]:

        if not os.path.exists(os.path.abspath(mod_path)):
            raise FileNotFoundError(f"Model path not found: {mod_path}")

    if not os.path.exists(os.path.abspath(args.json_file)):
        raise FileNotFoundError(f"JSON file not found: {args.json_file}")

    print("\033[94m\033[1mLoading models...\033[0m\033[0m")
    models = load_models(
        device,
        os.path.abspath(args.localisation_model_path),
        os.path.abspath(args.binary_model_path),
        os.path.abspath(args.order_model_path),
        os.path.abspath(args.order_thresholds_path),
        os.path.abspath(args.species_model_path),
        os.path.abspath(args.species_labels),
        verbose=args.verbose,
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
        binary_model=models["classification_model"],
        order_model=models["order_model"],
        order_labels=models["order_model_labels"],
        order_data_thresholds=models["order_model_thresholds"],
        species_model=models["species_model"],
        species_labels=models["species_model_labels"],
        device=device,
        top_n=args.top_n_species,
        csv_file=args.csv_file,
        skip_processed=args.skip_processed,
        verbose=args.verbose,
    )
