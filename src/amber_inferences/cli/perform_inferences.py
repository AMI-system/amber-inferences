#!/usr/bin/env python3

import argparse
import json
import os
import torch
import pandas as pd
from pathlib import Path

from amber_inferences.utils.config import load_credentials
from amber_inferences.utils.custom_models import load_models
from amber_inferences.utils.deployment_summary import deployment_data
from amber_inferences.utils.inference_scripts import (
    download_and_analyse,
    initialise_session,
)


def main(
    chunk_id,
    json_file,
    output_dir,
    bucket_name,
    credentials_file=Path("credentials.json"),
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
    csv_file=Path("results.csv"),
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
    json_file = Path(json_file)
    csv_file = Path(csv_file)
    output_dir = Path(output_dir)
    credentials_file = Path(credentials_file)

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
    if csv_file.exists() and skip_processed:
        already_processed = pd.read_csv(csv_file, low_memory=False)
        csv_keys = already_processed["image_path"].tolist()
        csv_keys = [os.path.basename(key) for key in csv_keys]

        # order already processed keys by the last modified time
        csv_keys = sorted(csv_keys)

        # remove the last key processed so that gets re-run
        csv_keys = csv_keys[:-1] if len(csv_keys) > 0 else csv_keys

        keys = [key for key in keys if os.path.basename(key) not in csv_keys]
        keys = sorted(keys)

        if len(csv_keys) > 0 and verbose:
            print(f"Skipping {len(csv_keys)} images previously processed.")

    # exit if length keys is 0
    if len(keys) == 0:
        print(f"All images already processed in {csv_file}")
        return

    credentials = load_credentials("./credentials.json")
    dep = keys[0].split("/")[0]
    dep_data = deployment_data(
        credentials,
        subset_countries=[bucket_name],
        subset_deployments=[dep],
        include_file_count=False,
    )[dep]

    download_and_analyse(
        keys=keys,
        output_dir=output_dir,
        dep_data=dep_data,
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


def check_model_paths(args):
    for mod_path, permitted_filetypes in [
        (args.localisation_model_path, [".pt", ".pth"]),
        (args.binary_model_path, [".pt", ".pth"]),
        (args.order_model_path, [".pt", ".pth"]),
        (args.order_thresholds_path, [".csv"]),
        (args.species_model_path, [".pt", ".pth"]),
        (args.species_labels, ["json"]),
    ]:
        if not mod_path.resolve().exists():
            raise FileNotFoundError(f"File not found: {mod_path}")
        if permitted_filetypes and not str(mod_path).endswith(
            tuple(permitted_filetypes)
        ):
            raise ValueError(
                f"File must be a {'|'.join(permitted_filetypes)}: {mod_path}"
            )
    if not args.json_file.resolve().exists():
        raise FileNotFoundError(f"JSON file not found: {args.json_file}")


def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Cuda available, using GPU")
    else:
        device = torch.device("cpu")
        print("Cuda not available, using CPU")
    return device


def validate_model_labels(models):
    if models["order_model"] is not None and models["order_model_labels"] is not None:
        model_out = models["order_model"]
        labels = models["order_model_labels"]
        if hasattr(model_out, "softmax_reg1") and hasattr(
            model_out.softmax_reg1, "out_features"
        ):
            n_model = model_out.softmax_reg1.out_features
            n_labels = len(labels)
            if n_model != n_labels:
                raise ValueError(
                    f"Order model output size ({n_model}) does not match number of order labels ({n_labels})"
                )
    if (
        models["species_model"] is not None
        and models["species_model_labels"] is not None
    ):
        model_out = models["species_model"]
        labels = models["species_model_labels"]
        if hasattr(model_out, "classifier") and hasattr(
            model_out.classifier, "out_features"
        ):
            n_model = model_out.classifier.out_features
            n_labels = len(labels)
            if n_model != n_labels:
                raise ValueError(
                    f"Species model output size ({n_model}) does not match number of species labels ({n_labels})"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific chunk of S3 keys.")
    parser.add_argument(
        "--chunk_id",
        required=True,
        help="ID of the chunk to process (e.g., 0, 1, 2, 3).",
    )
    parser.add_argument(
        "--json_file",
        required=True,
        help="Path to the JSON file with key chunks.",
        type=Path,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save downloaded files and analysis results.",
        default=Path("./data/"),
        type=Path,
    )
    parser.add_argument("--bucket_name", required=True, help="Name of the S3 bucket.")
    parser.add_argument(
        "--credentials_file",
        default=Path("credentials.json"),
        help="Path to AWS credentials file.",
        type=Path,
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
        type=Path,
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
        type=Path,
        help="Path to the binary model weights.",
        default=Path("./models/moth-nonmoth-effv2b3_20220506_061527_30.pth"),
    )
    parser.add_argument(
        "--order_model_path",
        type=Path,
        help="Path to the order model weights.",
        default=Path("./models/dhc_best_128.pth"),
    )
    parser.add_argument(
        "--order_labels", type=Path, help="Path to the order labels file."
    )
    parser.add_argument(
        "--species_model_path",
        type=Path,
        help="Path to the species model weights.",
        default=Path(
            "./models/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt"
        ),
    )
    parser.add_argument(
        "--species_labels",
        type=Path,
        help="Path to the species labels file.",
        default=Path("./models/03_costarica_data_category_map.json"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--order_thresholds_path",
        type=Path,
        help="Path to the order data thresholds file.",
        default=Path("./models/thresholdsTestTrain.csv"),
    )
    parser.add_argument(
        "--top_n_species",
        type=int,
        help="The number of predictions to output.",
        default=5,
    )
    parser.add_argument(
        "--csv_file",
        default=Path("results.csv"),
        help="Path to save analysis results.",
        type=Path,
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

    if args.skip_processed:
        print("Note: Images already processed will be skipped.")

    check_model_paths(args)
    device = select_device()
    print("Loading models...")
    try:
        models = load_models(
            device,
            args.localisation_model_path.resolve(),
            args.binary_model_path.resolve(),
            args.order_model_path.resolve(),
            args.order_thresholds_path.resolve(),
            args.species_model_path.resolve(),
            args.species_labels.resolve(),
            verbose=args.verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")
    validate_model_labels(models)
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
