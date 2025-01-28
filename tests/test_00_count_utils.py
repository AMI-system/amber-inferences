import unittest
from unittest.mock import MagicMock
from amber_inferences.utils.api_utils import count_files

class TestCountFiles(unittest.TestCase):

    def setUp(self):
        # Mock the S3 client
        self.s3_client = MagicMock()

    def test_count_files_success(self):
        # Mock the paginator to return two pages with 3 and 2 objects respectively
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"KeyCount": 3},
            {"KeyCount": 2}
        ]
        self.s3_client.get_paginator.return_value = paginator

        # Test the function
        result = count_files(self.s3_client, "test-bucket", "test/prefix")
        self.assertEqual(result, 5)

    def test_count_files_no_objects(self):
        # Mock the paginator to return an empty page
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]
        self.s3_client.get_paginator.return_value = paginator

        # Test the function
        result = count_files(self.s3_client, "test-bucket", "test/prefix")
        self.assertEqual(result, 0)
