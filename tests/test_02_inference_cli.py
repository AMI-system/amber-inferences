import unittest
import subprocess
import os

class TestCLI(unittest.TestCase):

    def test_help_message(self):
        """Test if the CLI displays the help message correctly."""
        result = subprocess.run(
            ["python3", "-m", "amber_inferences.cli.perform_inferences", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout)

    def test_missing_arguments(self):
        """Test if the CLI fails when required arguments are missing."""
        result = subprocess.run(
            ["python3", "-m", "amber_inferences.cli.perform_inferences"],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the following arguments are required", result.stderr)

    def test_valid_run(self):
        """Test if the CLI runs successfully with valid arguments."""
        result = subprocess.run(
            [
                "python3", "-m", "amber_inferences.cli.perform_inferences",
                "--chunk_id", "1",
                "--json_file", "./examples/dep000072_subset_keys.json",
                "--output_dir", "./data/",
                "--bucket_name", "gbr",
                "--credentials_file", "./credentials.json",
                "--csv_file", "./data/examples/dep000072.csv",
                "--species_model_path", "./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt",
                "--species_labels", "./models/03_uk_data_category_map.json",
                "--localisation_model_path", "./models/v1_localizmodel_2021-08-17-12-06.pt",
                "--perform_inference",
                "--remove_image"

            ],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)  # Ensure script runs successfully
        self.assertTrue(os.path.exists("./data/examples/dep000072.csv"))  # Check if output CSV exists


class TestPaths(unittest.TestCase):

    def setUp(self):
        """Setup: Define paths and cleanup from previous runs."""
        self.output_dir = "./data/"
        self.output_csv = os.path.join(self.output_dir, "dep000072.csv")

        # Ensure output files do not exist before test
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def test_invalid_json_file(self):
        """Test behavior when an invalid JSON file is provided."""
        result = subprocess.run(
                [
                    "python3", "-m", "amber_inferences.cli.perform_inferences",
                    "--chunk_id", "1",
                    "--json_file", "./examples/non_existent.json",
                    "--output_dir", "./data/",
                    "--bucket_name", "gbr",
                    "--credentials_file", "./credentials.json",
                    "--csv_file", "dep000072.csv",
                    "--species_model_path", "./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt",
                    "--species_labels", "./models/03_uk_data_category_map.json",
                    "--localisation_model_path", "./models/v1_localizmodel_2021-08-17-12-06.pt",
                    "--perform_inference",
                    "--remove_image",
                    "--save_crops"
                ],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)  # Expect failure
        self.assertIn("Error: JSON file not found", result.stderr)  # Adjust based on actual error message





if __name__ == "__main__":
    unittest.main()
