# utils/custom_models.py

import json

import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import SegmentationModel

add_safe_globals({"SegmentationModel": SegmentationModel})


class Resnet50_species(torch.nn.Module):
    """
    ResNet-50 model for species classification.

    Args:
        num_classes (int): Number of output classes.
    """

    def __init__(self, num_classes):
        """
        Initialise the ResNet-50 species classifier.

        Args:
            num_classes (int): Number of output classes.
        """
        super(Resnet50_species, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        out_dim = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        """
        Forward pass for the species classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ResNet50_order(nn.Module):
    """
    ResNet-50 model for order classification with optional CBAM and dropout.

    Args:
        use_cbam (bool): Whether to use CBAM (not implemented here).
        image_depth (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, use_cbam=True, image_depth=3, num_classes=20):
        """
        Initialise the ResNet-50 order classifier.

        Args:
            use_cbam (bool): Whether to use CBAM (not implemented here).
            image_depth (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super(ResNet50_order, self).__init__()

        self.expansion = 4
        self.out_channels = 512

        self.model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # pretrained=True) # 80.86, 25.6M
        # self.model_ft = models.resnet50(pretrained=True)

        # overwrite the 'fc' layer
        # print("In features", self.model_ft.fc.in_features)
        self.model_ft.fc = nn.Identity()  # Do nothing just pass input to output

        # At least one layer
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(
            self.out_channels * self.expansion, self.out_channels
        )
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        """
        Forward propagation of pretrained ResNet-50 for order classification.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.model_ft(x)

        x = self.drop(x)  # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        # level_1 = nn.Softmax(level_1)

        return level_1


def load_loc_model(weights_path, device):
    """
    Load a Faster R-CNN localisation model from weights.

    Args:
        weights_path (str or Path): Path to the model weights file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded localisation model.
    """
    if weights_path is None:
        return None

    # Load the localisation model
    model_loc = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model_loc.roi_heads.box_predictor.cls_score.in_features
    model_loc.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model_loc.load_state_dict(state_dict)
    model_loc = model_loc.to(device)
    model_loc.eval()
    return model_loc


def load_loc(localisation_model_path, device, verbose=False):
    """
    Load a localisation model, trying both Faster R-CNN and flatbug models.

    Args:
        localisation_model_path (str or Path): Path to the model weights file.
        device (torch.device): Device to load the model on.
        verbose (bool): Whether to print status messages.

    Returns:
        torch.nn.Module or flatbug.Predictor: Loaded localisation model, or None if loading fails.
    """
    if localisation_model_path is None:
        return None

    # Try loading the localisation model with load_loc_model first, then try flatbug
    try:
        localisation_model = load_loc_model(str(localisation_model_path), device)
        if verbose:
            print(
                f" - Loaded localisation model from: {localisation_model_path} ({type(localisation_model)})"
            )
    # if that failed, try loading the flatbug model
    except Exception:
        from flat_bug.predictor import Predictor

        try:
            localisation_model = Predictor(
                model=str(localisation_model_path), device=device, dtype="float16"
            )
            if verbose:
                print(
                    f" - Loaded localisation model from: {localisation_model_path} (flatbug)"
                )
        except Exception as f:
            print(
                f"Failed to load localisation model with load_loc_model or flat_bug: {f}"
            )
            localisation_model = None

    return localisation_model


def load_binary(binary_model_path, device, verbose=False):
    """
    Load a binary (moth/non-moth) classification model from weights.

    Args:
        binary_model_path (str or Path): Path to the model weights file.
        device (torch.device): Device to load the model on.
        verbose (bool): Whether to print status messages.

    Returns:
        torch.nn.Module: Loaded binary classification model.
    """
    if binary_model_path is None:
        return None

    classification_model = timm.create_model(
        "tf_efficientnetv2_b3", num_classes=2, weights=None
    )
    classification_model = classification_model.to(device)
    checkpoint = torch.load(
        str(binary_model_path), map_location=device, weights_only=True
    )
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    classification_model.load_state_dict(state_dict)
    classification_model.eval()
    if verbose:
        print(f" - Loaded binary model from: {binary_model_path}")

    return classification_model


def load_order(order_model_path, order_threshold_path, device, verbose=False):
    """
    Load an order classification model and its thresholds/labels from files.

    Args:
        order_model_path (str or Path): Path to the model weights file.
        order_threshold_path (str or Path): Path to the CSV file with thresholds and labels.
        device (torch.device): Device to load the model on.
        verbose (bool): Whether to print status messages.

    Returns:
        tuple: (model, order_data_thresholds, order_labels)
            model (torch.nn.Module): Loaded order model.
            order_data_thresholds (pd.DataFrame): DataFrame with thresholds.
            order_labels (list): List of class names.
    """
    if order_model_path is None:
        return None, None, None

    order_data_thresholds = pd.read_csv(order_threshold_path)
    order_labels = order_data_thresholds["ClassName"].to_list()
    num_classes = len(order_labels)
    model_order = ResNet50_order(num_classes=num_classes)
    model_order.load_state_dict(
        torch.load(str(order_model_path), map_location=device, weights_only=True)
    )
    model_order = model_order.to(device)
    model_order.eval()
    if verbose:
        print(f" - Loaded order model from: {order_model_path}")

    return model_order, order_data_thresholds, order_labels


def load_species(species_model_path, species_labels, device, verbose=False):
    """
    Load a species classification model and its label map from files.

    Args:
        species_model_path (str or Path): Path to the model weights file.
        species_labels (str or Path): Path to the JSON file with label mapping.
        device (torch.device): Device to load the model on.
        verbose (bool): Whether to print status messages.

    Returns:
        tuple: (species_model, species_category_map)
            species_model (torch.nn.Module): Loaded species model.
            species_category_map (dict): Mapping from label to index.
    """
    if species_model_path is None:
        return None, None

    species_category_map = json.load(open(species_labels))
    num_classes = len(species_category_map)
    species_model = Resnet50_species(num_classes=num_classes)
    species_model = species_model.to(device)
    checkpoint = torch.load(
        str(species_model_path), map_location=device, weights_only=True
    )
    # The model state dict is nested in some checkpoints, and not in others
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    species_model.load_state_dict(state_dict)
    species_model.eval()
    if verbose:
        print(f" - Loaded species model from: {species_model_path}")

    return species_model, species_category_map


def load_models(
    device,
    localisation_model_path=None,
    binary_model_path=None,
    order_model_path=None,
    order_threshold_path=None,
    species_model_path=None,
    species_labels=None,
    verbose=False,
):
    """
    Load all relevant models from their weight paths and return as a dictionary.

    Args:
        device (torch.device): Device to load models on.
        localisation_model_path (str or Path, optional): Path to localisation model weights.
        binary_model_path (str or Path, optional): Path to binary model weights.
        order_model_path (str or Path, optional): Path to order model weights.
        order_threshold_path (str or Path, optional): Path to order model thresholds CSV.
        species_model_path (str or Path, optional): Path to species model weights.
        species_labels (str or Path, optional): Path to species label map JSON.
        verbose (bool): Whether to print status messages.

    Returns:
        dict: Dictionary with loaded models and label/threshold data.
    """
    model_dict = {
        "localisation_model": None,
        "classification_model": None,
        "species_model": None,
        "species_model_labels": None,
        "order_model": None,
        "order_model_thresholds": None,
        "order_model_labels": None,
    }

    localisation_model = load_loc(localisation_model_path, device, verbose)
    model_dict["localisation_model"] = localisation_model

    # Load the binary model
    model_dict["classification_model"] = load_binary(binary_model_path, device, verbose)

    # Load the order model
    model_order_outputs = load_order(
        order_model_path, order_threshold_path, device, verbose
    )
    model_dict["order_model"] = model_order_outputs[0]
    model_dict["order_model_thresholds"] = model_order_outputs[1]
    model_dict["order_model_labels"] = model_order_outputs[2]

    # Load the species classifier model
    species_model_outputs = load_species(
        species_model_path, species_labels, device, verbose
    )
    model_dict["species_model"] = species_model_outputs[0]
    model_dict["species_model_labels"] = species_model_outputs[1]

    return model_dict
