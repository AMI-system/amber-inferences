import requests
import sys
import boto3


def get_buckets(s3_client):
    """Get a list of all S3 buckets."""
    response = s3_client.list_buckets()
    return [bucket["Name"] for bucket in response["Buckets"]]


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
    image_count = 0
    audio_count = 0
    other_count = 0
    other_file_types = []
    for page in page_iterator:
        # count += page.get("KeyCount", 0)

        # count files by type
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jpg"):
                image_count += 1
            elif obj["Key"].endswith(".wav"):
                audio_count += 1
            else:
                other_count += 1
                file_type = obj["Key"].split(".")[-1]
                if file_type not in other_file_types:
                    other_file_types.append(file_type)

    return {
        "image_count": image_count,
        "audio_count": audio_count,
        "other_count": other_count,
        "other_file_types": other_file_types,
    }


def deployments_summary(
    aws_credentials,
    include_inactive=False,
    subset_countries=None,
    subset_deployments=None,
    summary_subdir="snapshot_images",
    include_image_count=True,
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

    deployment_summary = {}

    for country in all_countries:
        country_depl = [x for x in all_deployments if x["country"].title() == country]

        if not country_depl:
            print(f"No deployments found for country: {country}")
            continue

        if subset_deployments:
            country_depl = [
                x for x in country_depl if x["deployment_id"] in subset_deployments
            ]

        for dep_info in country_depl:
            prefix = f"{dep_info['deployment_id']}/{summary_subdir}/"
            bucket_name = dep_info["country_code"].lower()

            if include_image_count:
                print(f'Counting files in {dep_info["deployment_id"]}...')
                counts = count_files(s3_client, bucket_name, prefix)
                dep_info["file_types"] = counts

            deployment_summary[dep_info["deployment_id"]] = dep_info

    return deployment_summary
