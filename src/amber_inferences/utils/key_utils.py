import json
import os
import pandas as pd
from math import ceil


def list_s3_keys(s3_client, bucket_name, deployment_id="", subdir=None):
    """
    List all keys in an S3 bucket under a specific prefix.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix to filter keys (default: "").

    Returns:
        list: A list of S3 object keys.
    """
    keys = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": deployment_id,
        }
        if subdir:
            list_kwargs["Prefix"] = os.path.join(deployment_id, subdir)
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

    # remove corrupt/hidden keys
    keys = [x for x in keys if not os.path.basename(x).startswith("$")]
    keys = [x for x in keys if not os.path.basename(x).startswith(".")]
    keys = [x for x in keys if "recycle" not in x]

    return keys


def save_keys(
    s3_client,
    bucket,
    deployment_id,
    output_file,
    subdir="snapshot_images",
    verbose=False,
):
    """
    Save S3 keys to a JSON file.

    Parameters:
        s3_client: Boto3 S3 client.
        bucket (str): Name of the S3 bucket.
        deployment_id (str): Deployment ID for filtering keys.
        output_file (str): Path to the output JSON file.
        subdir (str): Subdirectory to filter keys (default: "snapshot_images").
    """
    os.makedirs(os.path.dirname(output_file) or os.getcwd(), exist_ok=True)

    keys = list_s3_keys(s3_client, bucket, deployment_id, subdir)

    # sort keys
    keys.sort()

    # sort by session date
    df_json = pd.DataFrame(keys, columns=["filename"])
    df_json["datetime"] = df_json["filename"].apply(
        lambda x: x.split("/")[2].replace("-snapshot.jpg", "")
    )
    df_json["datetime"] = pd.to_datetime(df_json["datetime"], format="%Y%m%d%H%M%S")

    # if the time is < 12, add 24 hours to the date
    df_json["session"] = df_json["datetime"]
    df_json["session"] = df_json["datetime"].apply(
        lambda x: x - pd.Timedelta(days=1) if x.hour < 12 else x
    )
    df_json["session"] = df_json["session"].dt.strftime("%Y-%m-%d")

    sessions = df_json.groupby("session")["filename"].apply(list).to_dict()

    # Save keys to the output file
    if verbose:
        print(f"Saving all {len(keys)} keys/records to {output_file}")

    with open(output_file, "w", encoding="UTF-8") as f:
        json.dump(sessions, f, indent=4)


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
