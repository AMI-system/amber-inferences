from io import StringIO
import sys
import unittest
from unittest.mock import patch, MagicMock
from amber_inferences.utils.config import load_credentials
from amber_inferences.deployments import print_deployments

class TestPrintDeployments(unittest.TestCase):
    def setUp(self):
        self.aws_credentials = load_credentials("./credentials.json")

    @patch("amber_inferences.deployments.boto3.Session")
    @patch("amber_inferences.deployments.get_deployments")
    @patch("amber_inferences.deployments.count_files")
    def test_print_deployments_active_only(self, mock_count_files, mock_get_deployments, mock_boto_session):
        # Mock get_deployments to return both active and inactive deployments

        mock_get_deployments.return_value = [
            {"deployment_id": "dep000020", "location_name": "Loc1", "status": "active", "country": "Panama", "country_code": "pan"},
            {"deployment_id": "dep000022", "location_name": "Loc2", "status": "inactive", "country": "Panama", "country_code": "pan"},
        ]

        # Mock count_files to return a specific number
        mock_count_files.return_value = 100

        # Mock boto3 session and client
        mock_s3_client = MagicMock()
        mock_boto_session.return_value.client.return_value = mock_s3_client

        # Redirect stdout to capture print output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        print_deployments(self.aws_credentials, include_inactive=False, print_image_count=True)

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Assert the captured output contains expected strings
        output = captured_output.getvalue()
        self.assertIn("Deployment ID: dep000020 - Location: Loc1", output)
        self.assertIn(" - This deployment has 100 images.", output)
        self.assertNotIn("Deployment ID: dep000022 - Location: Loc2", output)

        # Verify count_files was called with the correct parameters
        mock_count_files.assert_called_with(mock_s3_client, "pan", "dep000020/snapshot_images")
