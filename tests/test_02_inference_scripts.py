import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from amber_inferences.utils.inference_scripts import flatbug, classify_species, classify_order, classify_box

class TestSpeciesScripts(unittest.TestCase):
    def test_classify_species(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.1, 0.4, 0.5, 0.2, 0.3]])
        mock_model.return_value = mock_output
        top_n = 3
        # Mock the category map
        regional_category_map = {"species_a": 0, "species_b": 1, "species_c":2 , "species_d": 3, "species_e": 4}

        predictions = torch.nn.functional.softmax(mock_output, dim=1).cpu().detach().numpy()[0]

        # Sort predictions to get the indices of the top 5 scores
        top_n_indices = predictions.argsort()[-top_n:][::-1]

        # Map indices to labels and fetch their confidence scores
        index_to_label = {index: label for label, index in regional_category_map.items()}
        top_n_labels = [index_to_label[idx] for idx in top_n_indices]
        top_n_scores = [predictions[idx] for idx in top_n_indices]

        # Call the function
        labels, scores = classify_species(torch.tensor([1, 2, 3]), mock_model, regional_category_map, top_n=top_n)

        # Check the results
        self.assertEqual(labels, top_n_labels)
        self.assertEqual(scores, top_n_scores)

class TestOrderScripts(unittest.TestCase):
    def test_classify_order(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = mock_output

        pred = torch.nn.functional.softmax(mock_output, dim=1)
        predictions = pred.cpu().detach().numpy()
        pred_score = predictions.max(axis=1).astype(float)[0]

        # Mock the labels and thresholds
        order_labels = ["order_a", "order_b"]
        order_data_thresholds = [0.5, 0.5]

        # Call the function
        label, score = classify_order(torch.tensor([1, 2, 3]), mock_model, order_labels, order_data_thresholds)

        # Check the results
        self.assertEqual(label, "order_b")
        self.assertEqual(score, pred_score)

class TestBoxScripts(unittest.TestCase):
    def test_classify_box(self):
        # Mock the model
        mock_model = MagicMock()
        mock_output = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictions = torch.nn.functional.softmax(mock_output, dim=1)
        predictions = predictions.cpu().detach().numpy()
        categories = predictions.argmax(axis=1)

        score_pred = predictions.max(axis=1).astype(float)[0]

        # Call the function
        label, score = classify_box(torch.tensor([1, 2, 3, 4, 5]), mock_model)


        # Check the results
        self.assertEqual(label, "nonmoth")
        self.assertEqual(score, score_pred)

