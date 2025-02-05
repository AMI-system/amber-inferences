import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from amber_inferences.utils.inference_scripts import flatbug, classify_species, classify_order, classify_box

class TestSpeciesScripts(unittest.TestCase):
    def test_classify_species(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_model.return_value = mock_output

        # Mock the category map
        regional_category_map = {0: "species_a", 1: "species_b", 2: "species_c", 3: "species_d", 4: "species_e"}

        # Call the function
        labels, scores = classify_species(torch.tensor([1, 2, 3]), mock_model, regional_category_map, top_n=3)

        # Check the results
        self.assertEqual(labels, ["species_e", "species_d", "species_c"])
        self.assertEqual(scores, [0.5, 0.4, 0.3])

class TestOrderScripts(unittest.TestCase):
    def test_classify_order(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = mock_output

        # Mock the labels and thresholds
        order_labels = ["order_a", "order_b"]
        order_data_thresholds = [0.5, 0.5]

        # Call the function
        label, score = classify_order(torch.tensor([1, 2, 3]), mock_model, order_labels, order_data_thresholds)

        # Check the results
        self.assertEqual(label, "order_b")
        self.assertEqual(score, 0.9)

class TestBoxScripts(unittest.TestCase):
    def test_classify_box(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        # Call the function
        label, score = classify_box(torch.tensor([1, 2, 3]), mock_model)

        # Check the results
        self.assertEqual(label, "nonmoth")
        self.assertEqual(score, 0.7)

class TestFlatBugScripts(unittest.TestCase):
    @patch("amber_inferences.utils.inference_scripts.os.path.dirname")
    @patch("amber_inferences.utils.inference_scripts.os.path.basename")
    def test_flatbug(self, mock_basename, mock_dirname):
        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value.json_data = {"boxes": [1, 2, 3]}

        # Mock the os.path functions
        mock_dirname.return_value = "/mock/dir"
        mock_basename.return_value = "image.jpg"

        # Call the function
        flatbug("/mock/dir/image.jpg", mock_model)

        # Check if the model was called with the correct argument
        mock_model.assert_called_with("/mock/dir/image.jpg")

        # Check if the plot method was called
        mock_model.return_value.plot.assert_called_with(outpath="/mock/dir/flatbug/flatbug_image.jpg")

if __name__ == "__main__":
    unittest.main()