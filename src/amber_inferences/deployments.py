import boto3
from .utils.api_utils import get_deployments, count_files


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
