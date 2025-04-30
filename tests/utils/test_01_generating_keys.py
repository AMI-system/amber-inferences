import unittest
from unittest.mock import patch, mock_open, MagicMock
from amber_inferences.utils.key_utils import list_s3_keys, save_keys


class TestGeneratingKeys(unittest.TestCase):

    def test_list_s3_keys(self):
        # Create a mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.side_effect = [
            {
                "Contents": [{"Key": "file1.jpg"}, {"Key": "file2.jpg"}],
                "IsTruncated": False,
            }
        ]

        # Call the function
        keys = list_s3_keys(mock_s3_client, "test-bucket", deployment_id="test-prefix")

        # Assertions
        self.assertEqual(keys, ["file1.jpg", "file2.jpg"])
        mock_s3_client.list_objects_v2.assert_called()

    @patch("amber_inferences.utils.key_utils.list_s3_keys")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")  # to avoid creating real directories
    def test_save_keys(self, mock_makedirs, mock_open_file, mock_list_s3_keys):
        # Setup
        mock_list_s3_keys.return_value = ["file1.jpg", "file2.jpg"]
        mock_s3_client = MagicMock()
        output_file = "test_output.json"

        # Call the function
        save_keys(mock_s3_client, "test-bucket", "test-deployment", output_file)

        # Assertions
        mock_list_s3_keys.assert_called_once_with(
            mock_s3_client, "test-bucket", "test-deployment", "snapshot_images"
        )
        mock_makedirs.assert_called()
        mock_open_file.assert_called_once_with(output_file, "w", encoding="UTF-8")

        # Check that the file was written with correct JSON
        handle = mock_open_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn('"file1.jpg"', written_data)
        self.assertIn('"file2.jpg"', written_data)


if __name__ == "__main__":
    unittest.main()
