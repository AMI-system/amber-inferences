import unittest
from unittest.mock import patch, Mock
from amber_inferences.utils.api_utils import get_deployments


class TestGetDeployments(unittest.TestCase):

    # test a successful API call
    @patch("amber_inferences.utils.api_utils.requests.get")
    def test_get_deployments_success(self, mock_get):
        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "deployment_id": "123",
                "country": "UK",
                "status": "active",
                "country_code": "GB",
            }
        ]
        mock_get.return_value = mock_response

        # Test the function
        result = get_deployments("username", "password")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["deployment_id"], "123")

    # # test a failed API call
    # @patch("amber_inferences.utils.api_utils.requests.get")
    # @patch("sys.exit")  # Mock sys.exit() to prevent it from terminating the test
    # def test_get_deployments_failure(self, mock_exit, mock_get):
    #     # Mock a failed HTTP request (401 Unauthorized)
    #     mock_response = Mock()
    #     mock_response.status_code = 401
    #     # Simulate HTTPError
    #     mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")

    #     mock_get.return_value = mock_response

    #     # Now, test that sys.exit(1) is called and capture the SystemExit exception
    #     raised = False
    #     try:
    #         print(get_deployments("wrong_user", "wrong_pass"))  # This should trigger sys.exit
    #         print("A")
    #     except SystemExit as e:
    #         print("B")
    #         raised = True
    #         self.assertEqual(e.code, 1)  # Ensure the exit code is 1

    #     self.assertTrue(raised)  # Ensure that the exception was raised
    #     mock_exit.assert_called_once_with(1)  # Ensu
