from io import StringIO
import sys
import unittest
from unittest.mock import patch, MagicMock
from amber_inferences.utils.config import load_credentials
from amber_inferences.deployments import print_deployments, count_files

class TestPrintDeployments(unittest.TestCase):
    def setUp(self):
        self.aws_credentials = load_credentials("./credentials.json")

    @patch("amber_inferences.deployments.boto3.Session")
    @patch("amber_inferences.deployments.get_deployments")
    @patch("amber_inferences.deployments.count_files")
    def test_print_deployments_active_only(self, mock_count_files, mock_get_deployments, mock_boto_session):
        # Mock get_deployments to return both active and inactive deployments

        mock_get_deployments.return_value = [
            {"deployment_id": "dep000020", "location_name": "trap_4", "status": "active", "country": "Panama", "country_code": "pan"},
            {"deployment_id": "dep000022", "location_name": "trap_6", "status": "inactive", "country": "Panama", "country_code": "pan"},
        ]

        # Mock count_files to return a specific number
        mock_count_files.return_value = 0

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
        self.assertIn("Deployment ID: dep000020 - Location: trap_4", output)
        self.assertIn(" - This deployment has 0 images.", output)
        self.assertNotIn("Deployment ID: fake_deployment - Location: fake_deployment", output)

    def test_count_files(self):
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_paginator = mock_s3_client.get_paginator.return_value

        # Mock paginator's paginate method to return fake S3 response
        mock_paginator.paginate.return_value = [
            {"KeyCount": 5},
            {"KeyCount": 3},
            {"KeyCount": 2}
        ]

        # Call count_files function
        result = count_files(mock_s3_client, "test-bucket", "test-prefix")

        # Assert that the total count is correct (5 + 3 + 2 = 10)
        self.assertEqual(result, 10)

        # Ensure that list_objects_v2 paginator was called with the correct parameters
        mock_s3_client.get_paginator.assert_called_with("list_objects_v2")
        mock_paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="test-prefix")