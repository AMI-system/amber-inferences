import unittest
from unittest.mock import patch
from amber_inferences.utils.config import load_credentials


class TestLoadCredentials(unittest.TestCase):

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_credentials_file_not_found(self, mock_file):
        # Test loading the credentials file when it does not exist
        with self.assertRaises(FileNotFoundError):
            load_credentials("./non_existent_file.json")

    # test the credentials has required keys
    def test_aws_credentials_structure_success(self):
        # Expected keys
        expected_keys = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_URL_ENDPOINT",
        }
        aws_credentials = load_credentials("./credentials.json")

        # Check if all required keys are present
        self.assertTrue(
            expected_keys.issubset(aws_credentials.keys()),
            "Missing required keys in AWS credentials.",
        )

        # Check if all values are non-empty strings
        for key in expected_keys:
            self.assertIsInstance(aws_credentials[key], str, f"{key} must be a string.")
            self.assertNotEqual(aws_credentials[key], "", f"{key} cannot be empty.")
