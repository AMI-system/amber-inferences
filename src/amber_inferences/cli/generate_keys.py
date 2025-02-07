#!/usr/bin/env python3

import argparse
import os

from amber_inferences.utils.key_utils import (
    list_s3_keys,
    load_workload,
    save_chunks,
    save_keys_to_file,
    split_workload,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a file containing S3 keys from a bucket."
    )
    parser.add_argument(
        "--bucket", type=str, required=True, help="Name of the S3 bucket."
    )
    parser.add_argument(
        "--deployment_id",
        type=str,
        default="",
        help="The deployment id to filter objects. If set to '' then all deployments are used. (default: '')",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="s3_keys.txt",
        help="Output file to save S3 keys.",
    )
    parser.add_argument(
        "--file_extensions",
        type=str,
        nargs="+",
        default="'jpg' 'jpeg'",
        help="File extensions to be included. If empty, all extensions used. Defauly = 'jpg' 'jpeg'",
    )
    parser.add_argument(
        "--subset_dates",
        type=str,
        nargs="+",
        default=None,
        help="Dates to subset the keys. If empty, all dates used. Default = None. Format: 'YYYY-MM-DD'",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100, help="Number of keys per chunk."
    )
    args = parser.parse_args()

    # List keys from the specified S3 bucket and prefix
    print(
        f"Listing keys from bucket '{args.bucket}' with deployment '{args.deployment_id}'..."
    )
    keys = list_s3_keys(args.bucket, args.deployment_id)

    # Save keys to the output file
    all_records_file = (
        f"{os.path.dirname(args.output_file)}/"
        f"{os.path.splitext(os.path.basename(args.output_file))[0]}"
        "_all_keys"
        f"{os.path.splitext(os.path.basename(args.output_file))[1]}"
    )
    save_keys_to_file(keys, all_records_file)
    print(f"Saving all {len(keys)} keys/records to {all_records_file}")

    # Load the workload from the input file
    keys = load_workload(all_records_file, args.file_extensions, args.subset_dates)

    # Split the workload into chunks
    chunks = split_workload(keys, args.chunk_size)

    # Save the chunks to a JSON file
    save_chunks(chunks, args.output_file)

    print(f"Successfully split {len(keys)} keys into {len(chunks)} chunks.")
    print(f"Chunks saved to {args.output_file}")


if __name__ == "__main__":
    main()

    parser = argparse.ArgumentParser(
        description="Pre-chop S3 workload into manageable chunks."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to file containing S3 keys, one per line.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSON file.",
    )
    args = parser.parse_args()
