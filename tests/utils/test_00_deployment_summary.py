import unittest
from unittest.mock import MagicMock, patch
from amber_inferences.utils.deployment_summary import count_files


class TestCountFiles(unittest.TestCase):
    @patch("amber_inferences.utils.deployment_summary.requests.get")
    def test_count_files_with_images_and_audio(self, mock_get):
        # Mock the requests.get so no real HTTP call happens
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = []

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


if __name__ == "__main__":
    unittest.main()
