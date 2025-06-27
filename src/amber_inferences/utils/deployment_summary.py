#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously and performs
inference on the images. AWS credentials, S3 bucket name, and UKCEH API
credentials are loaded from a configuration file (credentials.json).
"""

import argparse
import json
import boto3
import sys
import requests


# TODO: archive, used in api_utils
def get_deployments(username, password):
    """Fetch deployments from the API with authentication."""

    try:
        url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
        response = requests.post(
            url, data={"username": username, "password": password}, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        if response.status_code == 404:
            print(
                "App not available. Check at https://connect-apps.ceh.ac.uk/ami-data-upload/"
            )
        if response.status_code == 401:
            print("Wrong username or password. Try again!")
        sys.exit(1)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)


def count_files(s3_client, bucket_name, prefix):
    """
    Count number of files for a given prefix.
    """
    image_count = 0
    audio_count = 0

    keys = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": prefix,
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

    image_count = len([x for x in keys if x.endswith(".jpg")])
    audio_count = len([x for x in keys if x.endswith(".wav")])

    return {"keys": keys, "image_count": image_count, "audio_count": audio_count}

    # for page in page_iterator:

    # return {'image_count': image_count, 'audio_count': audio_count}


def deployment_data(
    credentials,
    include_inactive=False,
    subset_countries=None,
    subset_deployments=None,
    include_file_count=True,
):
    """
    Provide the deployments available through the object store.
    """

    # Get all deployments
    username = credentials["UKCEH_username"]
    password = credentials["UKCEH_password"]
    all_deployments = get_deployments(username, password)

    session = boto3.Session(
        aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=credentials["AWS_URL_ENDPOINT"])

    if not include_inactive:
        act_string = " active "
        all_deployments = [x for x in all_deployments if x["status"] == "active"]
    else:
        act_string = " "

    # Loop through each country to print deployment information
    all_countries = list(set([dep["country"].title() for dep in all_deployments]))
    all_ccodes = list(set([dep["country_code"].title() for dep in all_deployments]))

    if subset_countries is not None:
        subset_countries = [x.title() for x in subset_countries]
        not_included_countries = [
            x
            for x in subset_countries
            if (x not in all_countries) and (x not in all_ccodes)
        ]
        for missing in not_included_countries:
            print(
                f"\033[1mWARNING: {missing} does not have any {act_string}deployments, check spelling\033[0m"
            )
        all_countries = [x for x in all_countries if x in subset_countries]

        # if country code was used
        all_countries = all_countries + list(
            set(
                [
                    x["country"].title()
                    for x in all_deployments
                    if x["country_code"].title() in subset_countries
                ]
            )
        )

    deployment_data = {}
    for country in all_countries:
        country_depl = [x for x in all_deployments if x["country"].title() == country]
        country_code = list(set([x["country_code"] for x in country_depl]))[0]
        all_deps = list(set([x["deployment_id"] for x in country_depl]))

        if subset_deployments is not None:
            # catch if subset_deployments are not in all_deps
            subset_deployments_true = [x for x in subset_deployments if x in all_deps]
            # print the dropped deployments
            if len(subset_deployments_true) < len(subset_deployments):
                dropped_deps = [
                    x for x in subset_deployments if x not in subset_deployments_true
                ]
                print(
                    (
                        f"\033[1mWARNING: The following deployments were not found for {country}:"
                        f" {', '.join(dropped_deps)}\033[0m"
                    )
                )

            if not subset_deployments:
                print(
                    f"\033[1mWARNING: No deployments found for {country} with the specified subset.\033[0m"
                )
                continue
            all_deps = [x for x in all_deps if x in subset_deployments_true]

        for dep in sorted(all_deps):
            dep_info = [x for x in country_depl if x["deployment_id"] == dep][0]

            # get the number of files for this deployment
            prefix = f"{dep_info['deployment_id']}/snapshot_images"
            bucket_name = dep_info["country_code"].lower()
            dep_info["bucket_name"] = bucket_name
            dep_info["prefix"] = prefix
            dep_info["country"] = country
            dep_info["country_code"] = country_code

            if include_file_count:
                counts = count_files(s3_client, bucket_name, prefix)
                dep_info["image_count"] = counts["image_count"]
                dep_info["audio_count"] = counts["audio_count"]

            deployment_data[dep] = dep_info
    return deployment_data


def print_deployments(
    credentials, include_inactive=False, subset_countries=None, print_file_count=True
):
    """
    Provide the deployments available through the object store.
    """

    # Get all deployments
    username = credentials["UKCEH_username"]
    password = credentials["UKCEH_password"]
    all_deployments = get_deployments(username, password)

    session = boto3.Session(
        aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=credentials["AWS_URL_ENDPOINT"])

    if not include_inactive:
        act_string = " active "
        all_deployments = [x for x in all_deployments if x["status"] == "active"]
    else:
        act_string = " "

    # Loop through each country to print deployment information
    all_countries = list(set([dep["country"].title() for dep in all_deployments]))

    if subset_countries is not None:
        subset_countries = [x.title() for x in subset_countries]
        not_included_countries = [x for x in subset_countries if x not in all_countries]
        for missing in not_included_countries:
            print(
                f"\033[1mWARNING: {missing} does not have any {act_string}deployments, check spelling\033[0m"
            )
        all_countries = [x for x in all_countries if x in subset_countries]

    for country in all_countries:
        country_depl = [x for x in all_deployments if x["country"].title() == country]
        country_code = list(set([x["country_code"] for x in country_depl]))[0]
        print(
            "\n\033[1m"
            + country
            + " ("
            + country_code
            + ") has "
            + str(len(country_depl))
            + act_string
            + "deployments:\033[0m"
        )
        all_deps = list(set([x["deployment_id"] for x in country_depl]))

        total_images = 0
        total_audio = 0
        for dep in sorted(all_deps):
            dep_info = [x for x in country_depl if x["deployment_id"] == dep][0]
            print(
                f"\033[1m - Deployment ID: {dep_info['deployment_id']}"
                + f", Name: {dep_info['location_name']}\033[0m"
            )

            # get the number of images for this deployment
            prefix = f"{dep_info['deployment_id']}/snapshot_images"
            bucket_name = dep_info["country_code"].lower()

            if print_file_count:
                counts = count_files(s3_client, bucket_name, prefix)
                print(counts)
                total_images = total_images + counts["image_count"]
                total_audio = total_audio + counts["audio_count"]
                print(
                    f" - This deployment has \033[1m{counts['image_count']}\033[0m"
                    + f" images and \033[1m{counts['audio_count']}\033[0m audio files.\n"
                )

        if print_file_count:
            print(
                f"\033[1m - {country} has {total_images}\033[0m images total\033[0m\n"
            )
            print(
                f"\033[1m - {country} has {total_audio}\033[0m audio files total\033[0m\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Script for printing the deployments available on the Jasmin object store."
    )
    parser.add_argument(
        "--credentials_file",
        default="./credentials.json",
        help="Path ot the aws credentials.",
    )
    parser.add_argument(
        "--include_inactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to include inactive deployments.",
    )
    parser.add_argument(
        "--print_file_count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to print the number of images per deployment.",
    )
    parser.add_argument(
        "--subset_countries",
        nargs="+",
        help="Optional list to subset for specific countries using name/code(e.g. --subset_countries 'Panama' 'tha').",
        default=None,
    )
    args = parser.parse_args()

    with open(args.credentials_file, encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)

    print_deployments(
        aws_credentials,
        args.include_inactive,
        args.subset_countries,
        args.print_file_count,
    )


if __name__ == "__main__":
    main()
