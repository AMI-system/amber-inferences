#!/usr/bin/env python3

import argparse
import boto3
from amber_inferences.utils.config import load_credentials
from amber_inferences.utils.key_utils import save_keys


def main():
    """
    Main function to generate a file containing S3 keys from a specified bucket.
    It lists all keys in the bucket, optionally filtered by deployment ID and file extensions,
    and saves them to an output file.
    """
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
    aws_credentials = load_credentials()
    # Setup boto3 session
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    # List keys from the specified S3 bucket and prefix
    print(
        f"Listing keys from bucket '{args.bucket}' with deployment '{args.deployment_id}'..."
    )

    save_keys(
        s3_client, args.bucket, args.deployment_id, args.output_file, verbose=True
    )


if __name__ == "__main__":
    main()
