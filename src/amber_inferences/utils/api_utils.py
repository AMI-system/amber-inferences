import requests
from requests.auth import HTTPBasicAuth
import sys


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