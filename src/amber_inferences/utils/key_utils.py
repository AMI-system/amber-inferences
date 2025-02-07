import json
import os
from math import ceil

import boto3


def list_s3_keys(bucket_name, deployment_id=""):
    """
    List all keys in an S3 bucket under a specific prefix.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix to filter keys (default: "").

    Returns:
        list: A list of S3 object keys.
    """
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)

    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    keys = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": deployment_id,
        }
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        # Add object keys to the list
        for obj in response.get("Contents", []):
            keys.append(obj["Key"])

        # Check if there are more objects to list
        if response.get("IsTruncated"):  # If True, there are more results
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return keys


def save_keys_to_file(keys, output_file):
    """
    Save S3 keys to a file, one per line.

    Parameters:
        keys (list): List of S3 keys.
        output_file (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(output_file) or os.getcwd(), exist_ok=True)

    with open(output_file, "w") as f:
        for key in keys:
            f.write(key + "\n")


def load_workload(input_file, file_extensions, subset_dates=None):
    """
    Load workload from a file. Assumes each line contains an S3 key.
    """
    with open(input_file, "r", encoding="UTF-8") as f:
        all_keys = [line.strip() for line in f.readlines()]

    subset_keys = [x for x in all_keys if x.endswith(tuple(file_extensions))]

    # subset by dates
    if subset_dates:
        subset_dates = [date.replace("-", "") for date in subset_dates]
        subset_keys = [
            x for x in subset_keys if any(date in x for date in subset_dates)
        ]

        if len(subset_keys) == 0:
            print(
                "No keys found for the specified dates. Please check the date format (YYYY-MM-DD) and try again."
            )

    # remove corrupt keys
    subset_keys = [x for x in subset_keys if not os.path.basename(x).startswith("$")]
    subset_keys = [x for x in subset_keys if not os.path.basename(x).startswith(".")]

    # remove keys uploaded from the recycle bin (legacy code)
    subset_keys = [x for x in subset_keys if "recycle" not in x]
    print(f"{len(subset_keys)} keys")

    return subset_keys


def split_workload(keys, chunk_size):
    """
    Split a list of keys into chunks of a specified size.
    """
    num_chunks = ceil(len(keys) / chunk_size)
    chunks = {
        str(i + 1): {"keys": keys[i * chunk_size : (i + 1) * chunk_size]}
        for i in range(num_chunks)
    }
    return chunks


def save_chunks(chunks, output_file):
    """
    Save chunks to a JSON file.
    """
    # create dir if not existant
    os.makedirs(os.path.dirname(output_file) or os.getcwd(), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=4)
