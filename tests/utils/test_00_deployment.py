from io import StringIO
import unittest
from unittest.mock import patch, MagicMock
from amber_inferences.utils.config import load_credentials
from amber_inferences.utils.deployment_summary import count_files, print_deployments


class TestPrintDeployments(unittest.TestCase):
    def setUp(self):
        self.aws_credentials = load_credentials("./credentials.json")

    @patch("amber_inferences.utils.deployment_summary.boto3.Session")
    @patch("amber_inferences.utils.deployment_summary.get_deployments")
    @patch("amber_inferences.utils.deployment_summary.count_files")
    def test_print_deployments_active_only(
        self, mock_count_files, mock_get_deployments, mock_boto_session
    ):
        # Mock get_deployments to return active and inactive deployments
        mock_get_deployments.return_value = [
            {
                "deployment_id": "dep000020",
                "location_name": "trap_4",
                "status": "active",
                "country": "Panama",
                "country_code": "pan",
            },
            {
                "deployment_id": "dep000022",
                "location_name": "trap_6",
                "status": "inactive",
                "country": "Panama",
                "country_code": "pan",
            },
        ]

        # Mock count_files to return a dictionary as expected
        mock_count_files.return_value = {"keys": [], "image_count": 0, "audio_count": 0}

        # Mock boto3 session and client
        mock_s3_client = MagicMock()
        mock_boto_session.return_value.client.return_value = mock_s3_client

        with patch("sys.stdout", new=StringIO()) as fake_out:
            print_deployments(
                self.aws_credentials, include_inactive=False, print_file_count=True
            )
            output = fake_out.getvalue()

        # Assert the output contains active deployment
        self.assertIn("Deployment ID: dep000020", output)
        self.assertIn(
            " - This deployment has \033[1m0\033[0m images and \033[1m0\033[0m audio files.",
            output,
        )

        # Assert the inactive deployment is not printed
        self.assertNotIn("dep000022", output)


class TestCountFiles(unittest.TestCase):
    def test_count_files_with_images_and_audio(self):
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "someprefix/image1.jpg"},
                {"Key": "someprefix/sound1.wav"},
                {"Key": "someprefix/image2.jpg"},
            ],
            "IsTruncated": False,
        }

        result = count_files(mock_s3_client, "bucket", "someprefix/")
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["audio_count"], 1)
