import requests
from requests.auth import HTTPBasicAuth
import sys
import boto3


def get_deployments(username, password):
    """Fetch deployments from the API with authentication."""
    try:
        url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
        response = requests.get(
            url, auth=HTTPBasicAuth(username, password), timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        if response.status_code == 401:
            print("Wrong username or password. Try again!")
            sys.exit(1)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)

def count_files(s3_client, bucket_name, prefix):
    """Count number of files in a given S3 bucket with a prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)
    count = 0
    for page in page_iterator:
        count += page.get("KeyCount", 0)
    return count

def print_deployments(
    aws_credentials,
    include_inactive=False,
    subset_countries=None,
    print_image_count=True
):
    """Print information about deployments from the API."""
    # Setup boto3 session
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    # Get deployments
    username = aws_credentials["UKCEH_username"]
    password = aws_credentials["UKCEH_password"]
    all_deployments = get_deployments(username, password)

    if not include_inactive:
        all_deployments = [x for x in all_deployments if x["status"] == "active"]

    all_countries = list(set([dep["country"].title() for dep in all_deployments]))

    if subset_countries:
        subset_countries = [x.title() for x in subset_countries]
        all_countries = [x for x in all_countries if x in subset_countries]

    for country in all_countries:
        country_depl = [x for x in all_deployments if x["country"] == country]

        if not country_depl:
            print(f"No deployments found for country: {country}")
            continue

        country_code = list(set([x["country_code"] for x in country_depl]))[0]
        total_images = 0

        for dep_info in country_depl:
            print(f"Deployment ID: {dep_info['deployment_id']} - Location: {dep_info['location_name']}")
            prefix = f"{dep_info['deployment_id']}/snapshot_images"
            bucket_name = dep_info["country_code"].lower()

            if print_image_count:
                count = count_files(s3_client, bucket_name, prefix)
                total_images += count
                print(f" - This deployment has {count} images.")

        if print_image_count:
            print(f"{country} has {total_images} images in total.")

