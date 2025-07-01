import requests
import sys


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


def get_deployment_names(username, password, bucket):
    response = get_deployments(username, password)
    response = [x for x in response if x["country_code"].lower() == bucket.lower()]
    deployment_names = [x["deployment_id"] for x in response]
    if not deployment_names:
        print(f"No deployments found for bucket: {bucket}")
        return []
    return deployment_names


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
