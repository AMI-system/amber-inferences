#!/usr/bin/env python3

import argparse
from amber_inferences.utils.config import load_credentials
import amber_inferences.utils.deployment_summary as deployment_summary


def main():
    parser = argparse.ArgumentParser(
        description="Script for printing deployments available on the Jasmin object store."
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
        help="Flag to print the number of files per deployment.",
    )
    parser.add_argument(
        "--subset_countries",
        nargs="+",
        help="Optional list to subset for specific countries.",
        default=None,
    )
    parser.add_argument(
        "--subset_deployments",
        nargs="+",
        help="Optional list to subset for specific deployments.",
        default=None,
    )
    args = parser.parse_args()

    # Load AWS credentials
    aws_credentials = load_credentials()

    # Print deployments
    summary = deployment_summary.deployment_data(
        aws_credentials,
        include_inactive=args.include_inactive,
        subset_countries=args.subset_countries,
        subset_deployments=args.subset_deployments,
        include_file_count=args.print_file_count,
    )

    for dep_info in summary.values():
        print(
            f"Deployment ID: {dep_info['deployment_id']} - Location: {dep_info['location_name']}"
        )

        if "n_images" in dep_info.keys():
            print(f" - This deployment has {dep_info['n_images']} images.")


if __name__ == "__main__":
    main()
