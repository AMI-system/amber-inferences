import unittest
from unittest.mock import patch, mock_open, MagicMock
from amber_inferences.utils.key_utils import list_s3_keys, save_keys_to_file

class TestGeneratingKeys(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data='{"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_REGION": "test", "AWS_URL_ENDPOINT": "http://test"}')
    @patch("boto3.Session.client")
    def test_list_s3_keys(self, mock_boto_client, mock_open_file):
        # Mock S3 client response
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "file1.jpg"}, {"Key": "file2.jpg"}], "IsTruncated": False}
        ]
        mock_boto_client.return_value = mock_s3

        # Call function
        keys = list_s3_keys("test-bucket", "test-prefix")

        # Assertions
        self.assertEqual(keys, ["file1.jpg", "file2.jpg"])
        mock_boto_client.assert_called_once_with("s3", endpoint_url="http://test")
        mock_open_file.assert_called_once_with("./credentials.json", encoding="utf-8")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_keys_to_file(self, mock_open_file):
        keys = ["file1.jpg", "file2.jpg"]
        output_file = "test_output.txt"

        # Call function
        save_keys_to_file(keys, output_file)

        # Assertions
        mock_open_file.assert_called_once_with(output_file, "w")
        mock_open_file().write.assert_any_call("file1.jpg\n")
        mock_open_file().write.assert_any_call("file2.jpg\n")
