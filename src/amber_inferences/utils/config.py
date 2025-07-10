import json
import os


def validate_aws_credentials(credentials):
    """
    Validate that the provided credentials dictionary contains all required AWS and UKCEH keys.

    Args:
        credentials (dict): Dictionary of credentials to validate.

    Raises:
        ValueError: If any required key is missing or has an invalid value.
    """
    required_keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_URL_ENDPOINT",
        "UKCEH_username",
        "UKCEH_password",
    ]
    missing_keys = list(set(required_keys) - set(list(credentials.keys())))
    if missing_keys:
        raise ValueError(f"Missing required credentials: {', '.join(missing_keys)}")
    for key in required_keys:
        if not isinstance(credentials[key], str) or not credentials[key]:
            raise ValueError(f"Credential {key} must be a non-empty string.")


def load_credentials(credentials_path="./credentials.json"):
    """
    Load AWS and UKCEH credentials from a JSON file and validate them.

    Args:
        credentials_path (str): Path to the credentials JSON file.

    Returns:
        dict: Dictionary of validated credentials.

    Raises:
        FileNotFoundError: If the credentials file does not exist.
        ValueError: If required credentials are missing or invalid.
    """
    if os.path.exists(credentials_path):
        with open(credentials_path, encoding="utf-8") as config_file:
            credentials = json.load(config_file)
            validate_aws_credentials(credentials)
            return credentials
    else:
        raise FileNotFoundError(f"Configuration file {credentials_path} not found.")
