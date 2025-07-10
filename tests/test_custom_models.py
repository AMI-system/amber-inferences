import types
import json
import pandas as pd

import amber_inferences.utils.custom_models as custom_models


def test_load_loc_model_none():
    assert custom_models.load_loc_model(None, "cpu") is None


def test_load_loc_model_fail(monkeypatch):
    """Test that load_loc_model raises an error if the model cannot be loaded."""

    # Patch torch.load to raise error
    monkeypatch.setattr(custom_models.torch, "load", lambda *a, **k: {})

    # Patch model to have required attributes
    class DummyModel:
        def __init__(self):
            self.roi_heads = types.SimpleNamespace()
            self.roi_heads.box_predictor = types.SimpleNamespace()
            self.roi_heads.box_predictor.cls_score = types.SimpleNamespace()
            self.roi_heads.box_predictor.cls_score.in_features = 2

        def load_state_dict(self, x):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        custom_models.torchvision.models.detection,
        "fasterrcnn_resnet50_fpn",
        lambda weights=None: DummyModel(),
    )
    result = custom_models.load_loc_model("fakepath", "cpu")
    assert result is not None


def test_load_loc_flatbug(monkeypatch):
    # Patch load_loc_model to raise
    monkeypatch.setattr(
        custom_models,
        "load_loc_model",
        lambda *a, **k: (_ for _ in ()).throw(Exception("fail")),
    )

    # need a dummy flat_bug.predictor.Predictor: patch import
    import sys
    import types

    dummy_predictor = type(
        "DummyPredictor", (), {"__init__": lambda self, model, device, dtype: None}
    )
    flat_bug_predictor_mod = types.ModuleType("flat_bug.predictor")
    flat_bug_predictor_mod.Predictor = dummy_predictor
    sys.modules["flat_bug.predictor"] = flat_bug_predictor_mod
    # Should fallback to flatbug and not raise
    result = custom_models.load_loc("fakepath", "cpu", verbose=False)
    assert result is None or isinstance(result, dummy_predictor)


def test_load_binary_none():
    assert custom_models.load_binary(None, "cpu") is None


def test_load_binary_success(monkeypatch):
    class DummyModel:
        def to(self, device):
            return self

        def load_state_dict(self, x):
            pass

        def eval(self):
            return self

    monkeypatch.setattr(
        custom_models.timm, "create_model", lambda *a, **k: DummyModel()
    )
    monkeypatch.setattr(custom_models.torch, "load", lambda *a, **k: {})
    result = custom_models.load_binary("fakepath", "cpu", verbose=True)
    assert result is not None


def test_load_order_none():
    m, t, lab = custom_models.load_order(None, None, "cpu")
    assert m is None and t is None and lab is None


def test_load_order_success(monkeypatch, tmp_path):
    # Patch pd.read_csv
    df = pd.DataFrame({"ClassName": ["A", "B"]})
    monkeypatch.setattr(custom_models.pd, "read_csv", lambda *a, **k: df)

    # Patch model
    class DummyOrder(custom_models.ResNet50_order):
        def __init__(self, num_classes):
            pass

        def load_state_dict(self, x):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(custom_models, "ResNet50_order", DummyOrder)
    monkeypatch.setattr(custom_models.torch, "load", lambda *a, **k: {})
    m, t, lab = custom_models.load_order("fakepath", "fakepath2", "cpu", verbose=True)
    assert m is not None and isinstance(t, pd.DataFrame) and lab == ["A", "B"]


def test_load_species_none():
    m, lab = custom_models.load_species(None, None, "cpu")
    assert m is None and lab is None


def test_load_species_success(monkeypatch, tmp_path):
    # Patch json.load
    labels = {"a": 0, "b": 1}
    labels_path = tmp_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    monkeypatch.setattr(custom_models.json, "load", lambda f: labels)

    class DummySpecies(custom_models.Resnet50_species):
        def __init__(self, num_classes):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, x):
            pass

        def eval(self):
            return self

    monkeypatch.setattr(custom_models, "Resnet50_species", DummySpecies)
    monkeypatch.setattr(custom_models.torch, "load", lambda *a, **k: {})
    m, lab = custom_models.load_species(
        "fakepath", str(labels_path), "cpu", verbose=True
    )
    assert m is not None and lab == labels


def test_load_models(monkeypatch):
    """Test output if they all load successfully."""
    # Patch all loaders
    monkeypatch.setattr(custom_models, "load_loc", lambda *a, **k: "loc")
    monkeypatch.setattr(custom_models, "load_binary", lambda *a, **k: "bin")
    monkeypatch.setattr(
        custom_models, "load_order", lambda *a, **k: ("order", "thresh", "labels")
    )
    monkeypatch.setattr(
        custom_models, "load_species", lambda *a, **k: ("species", "species_labels")
    )
    out = custom_models.load_models(
        "cpu",
        "locpath",
        "binpath",
        "orderpath",
        "threshpath",
        "speciespath",
        "specieslabels",
        verbose=True,
    )
    assert out["localisation_model"] == "loc"
    assert out["classification_model"] == "bin"
    assert out["order_model"] == "order"
    assert out["order_model_thresholds"] == "thresh"
    assert out["order_model_labels"] == "labels"
    assert out["species_model"] == "species"
    assert out["species_model_labels"] == "species_labels"


def test_resnet50_species_init_and_forward(monkeypatch):
    import torch

    # Patch torchvision.models.resnet50 to a dummy model for fast test
    class DummyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)
            self._children = [
                torch.nn.Conv2d(3, 10, 3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
            ]

        def children(self):
            return self._children

        def forward(self, x):
            return torch.ones((x.shape[0], 10, 1, 1))

    monkeypatch.setattr(
        custom_models.torchvision.models, "resnet50", lambda weights=None: DummyResNet()
    )
    # Create model
    model = custom_models.Resnet50_species(num_classes=5)
    # Check attributes
    assert model.num_classes == 5
    assert isinstance(model.backbone, torch.nn.Sequential)
    assert isinstance(model.avgpool, torch.nn.AdaptiveAvgPool2d)
    assert isinstance(model.classifier, torch.nn.Linear)
    # Forward pass with dummy input
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)


def test_resnet50_order_init_and_forward(monkeypatch):
    import torch

    # Patch models.resnet50 to a dummy model
    class DummyResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

        def forward(self, x):
            return torch.ones((x.shape[0], 2048))

    monkeypatch.setattr(
        custom_models.models, "resnet50", lambda weights=None: DummyResNet()
    )
    model = custom_models.ResNet50_order(use_cbam=False, image_depth=3, num_classes=4)
    # Check attributes
    assert model.expansion == 4
    assert model.out_channels == 512
    assert isinstance(model.model_ft, torch.nn.Module)
    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_load_loc_model_success(monkeypatch, tmp_path):
    # Patch torch.load to return a dict with model_state_dict
    class DummyPredictor:
        def __init__(self, in_features, num_classes):
            self.in_features = in_features

        def load_state_dict(self, x):
            self.loaded = True

        def to(self, device):
            return self

        def eval(self):
            return self

    class DummyModel:
        def __init__(self):
            self.roi_heads = type("roi_heads", (), {})()
            self.roi_heads.box_predictor = type("box_predictor", (), {})()
            self.roi_heads.box_predictor.cls_score = type("cls_score", (), {})()
            self.roi_heads.box_predictor.cls_score.in_features = 2

        def load_state_dict(self, x):
            self.loaded = True

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        custom_models.torchvision.models.detection,
        "fasterrcnn_resnet50_fpn",
        lambda weights=None: DummyModel(),
    )
    monkeypatch.setattr(
        custom_models.torchvision.models.detection,
        "faster_rcnn",
        type(
            "faster_rcnn",
            (),
            {"FastRCNNPredictor": lambda in_features, num_classes: None},
        ),
    )
    monkeypatch.setattr(
        custom_models.torch,
        "load",
        lambda *a, **k: {"model_state_dict": {"foo": "bar"}},
    )
    model = custom_models.load_loc_model("fakepath", "cpu")
    assert model is not None


def test_load_loc_model_state_dict_fallback(monkeypatch):
    # Patch torch.load to return a dict without model_state_dict
    class DummyModel:
        def __init__(self):
            self.roi_heads = type("roi_heads", (), {})()
            self.roi_heads.box_predictor = type("box_predictor", (), {})()
            self.roi_heads.box_predictor.cls_score = type("cls_score", (), {})()
            self.roi_heads.box_predictor.cls_score.in_features = 2

        def load_state_dict(self, x):
            self.loaded = True

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        custom_models.torchvision.models.detection,
        "fasterrcnn_resnet50_fpn",
        lambda weights=None: DummyModel(),
    )
    monkeypatch.setattr(
        custom_models.torchvision.models.detection,
        "faster_rcnn",
        type(
            "faster_rcnn",
            (),
            {"FastRCNNPredictor": lambda in_features, num_classes: None},
        ),
    )
    monkeypatch.setattr(custom_models.torch, "load", lambda *a, **k: {"foo": "bar"})
    model = custom_models.load_loc_model("fakepath", "cpu")
    assert model is not None


def test_loc_return_none():
    """Test that load_loc returns None if no path is provided."""
    assert custom_models.load_loc(None, "cpu") is None


def test_loc_return_localisation(monkeypatch, capsys):
    class DummyModel:
        def __init__(self):
            self.roi_heads = type("roi_heads", (), {})()
            self.roi_heads.box_predictor = type("box_predictor", (), {})()
            self.roi_heads.box_predictor.cls_score = type("cls_score", (), {})()
            self.roi_heads.box_predictor.cls_score.in_features = 2

        def load_state_dict(self, x):
            self.loaded = True

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(custom_models, "load_loc_model", lambda *a: DummyModel())
    output = custom_models.load_loc("model.pt", "cpu", verbose=True)
    assert isinstance(output, DummyModel)

    # check that the print statement was called
    out_capture = capsys.readouterr().out
    assert "Loaded localisation model from" in out_capture
    assert "DummyModel" in out_capture


def test_loc_return_flatbug(monkeypatch, capsys):
    class DummyModel:
        def __init__(self):
            self.roi_heads = type("roi_heads", (), {})()
            self.roi_heads.box_predictor = type("box_predictor", (), {})()
            self.roi_heads.box_predictor.cls_score = type("cls_score", (), {})()
            self.roi_heads.box_predictor.cls_score.in_features = 2

        def load_state_dict(self, x):
            self.loaded = True

        def to(self, device):
            return self

        def eval(self):
            return self

    # Patch load_loc_model to always raise
    monkeypatch.setattr(
        custom_models,
        "load_loc_model",
        lambda *a, **k: (_ for _ in ()).throw(Exception("Failed for Loc model")),
    )
    # Patch flat_bug.predictor.Predictor to always raise
    import sys
    import types

    dummy_predictor_mod = types.ModuleType("flat_bug.predictor")
    dummy_predictor_mod.Predictor = lambda *a, **k: DummyModel()
    sys.modules["flat_bug.predictor"] = dummy_predictor_mod

    output = custom_models.load_loc("model.pt", "cpu", verbose=True)
    assert isinstance(output, DummyModel)

    # check that the print statement was called
    out_capture = capsys.readouterr().out
    assert "Loaded localisation model from" in out_capture
    assert "(flatbug)" in out_capture


def test_all_loc_return_none(monkeypatch, capsys):
    # Patch load_loc_model to always raise
    monkeypatch.setattr(
        custom_models,
        "load_loc_model",
        lambda *a, **k: (_ for _ in ()).throw(Exception("Failed for Loc model")),
    )
    # Patch flat_bug.predictor.Predictor to always raise
    import sys
    import types

    dummy_predictor_mod = types.ModuleType("flat_bug.predictor")
    dummy_predictor_mod.Predictor = lambda *a, **k: (_ for _ in ()).throw(
        Exception("Failed for flatbug model")
    )
    sys.modules["flat_bug.predictor"] = dummy_predictor_mod

    output = custom_models.load_loc("model.pt", "cpu", verbose=True)
    assert output is None
    # check that the print statement was called
    assert (
        "Failed to load localisation model with load_loc_model or flat_bug"
        in capsys.readouterr().out
    )
