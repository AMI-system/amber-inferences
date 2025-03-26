#!/usr/bin/env python3

import argparse

# import os
# import boto3
# from amber_inferences.utils.config import load_credentials

from amber_inferences.utils.key_utils import (
    list_s3_keys,
    save_keys,
    # load_workload,
    # split_workload,
    # save_chunks,
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
    args = parser.parse_args()

    # Load AWS credentials
    # aws_credentials = load_credentials()
    # Setup boto3 session
    # session = boto3.Session(
    #     aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    #     aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    #     region_name=aws_credentials["AWS_REGION"],
    # )
    # s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    # List keys from the specified S3 bucket and prefix
    print(
        f"Listing keys from bucket '{args.bucket}' with deployment '{args.deployment_id}'..."
    )
    keys = list_s3_keys(args.bucket, args.deployment_id)

    # Save keys to the output file
    print(f"Saving all {len(keys)} keys/records to {args.output_file}")
    save_keys(args.bucket, args.deployment_id, args.output_file)

    # # Load the workload from the input file
    # keys = load_workload(all_records_file, args.file_extensions)

    # # Split the workload into chunks
    # chunks = split_workload(keys, args.chunk_size)

    # # Save the chunks to a JSON file
    # save_chunks(chunks, args.output_file)

    # print(f"Successfully split {len(keys)} keys into {len(chunks)} chunks.")
    # print(f"Chunks saved to {args.output_file}")


if __name__ == "__main__":
    main()

    # parser = argparse.ArgumentParser(
    #     description="Save image files from S3 workload into file."
    # )
    # parser.add_argument(
    #     "--input_file",
    #     type=str,
    #     required=True,
    #     help="Path to file containing S3 keys, one per line.",
    # )

    # parser.add_argument(
    #     "--output_file",
    #     type=str,
    #     required=True,
    #     help="Path to save the output JSON file.",
    # )
    # args = parser.parse_args()
