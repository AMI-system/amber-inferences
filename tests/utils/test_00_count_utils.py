import unittest
from unittest.mock import MagicMock
from amber_inferences.utils.api_utils import count_files


class TestCountFiles(unittest.TestCase):

    def setUp(self):
        # Mock the S3 client
        self.s3_client = MagicMock()

    def test_count_files_success(self):
        # Mock the paginator to return pages with objects
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "test/prefix/file1.jpg"},
                    {"Key": "test/prefix/file2.wav"},
                    {"Key": "test/prefix/file3.txt"},
                ]
            },
            {
                "Contents": [
                    {"Key": "test/prefix/file4.jpg"},
                    {"Key": "test/prefix/file5.docx"},
                ]
            },
        ]
        self.s3_client.get_paginator.return_value = paginator

        # Test the function
        result = count_files(self.s3_client, "test-bucket", "test/prefix")

        expected = {
            "image_count": 2,
            "audio_count": 1,
            "other_count": 2,
            "other_file_types": ["txt", "docx"],
        }
        self.assertEqual(result, expected)

    def test_count_files_no_objects(self):
        # Mock the paginator to return an empty page
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]
        self.s3_client.get_paginator.return_value = paginator

        # Test the function
        result = count_files(self.s3_client, "test-bucket", "test/prefix")

        expected = {
            "image_count": 0,
            "audio_count": 0,
            "other_count": 0,
            "other_file_types": [],
        }
        self.assertEqual(result, expected)
