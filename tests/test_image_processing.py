from unittest import mock
from PIL import Image
import numpy as np
import io

import amber_inferences.utils.image_processing as image_processing


def test_image_annotation_draw(monkeypatch, tmp_path):
    """Test drawing boxes on an image with labels."""
    # Create a blank image
    img = Image.new("RGB", (100, 100), color="white")
    # Save to a temp file
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    # Define boxes
    boxes = [
        {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50, "label": "A"},
        {"x_min": 60, "y_min": 60, "x_max": 90, "y_max": 90},
    ]
    # Patch plt.imshow to avoid displaying it
    monkeypatch.setattr(image_processing, "plt", mock.Mock(imshow=lambda img: None))
    image_processing.image_annotation(str(img_path), img=img, boxes=boxes)
    # No assertion needed, just check the is no error


def test_variance_of_laplacian(monkeypatch):
    """Test variance of Laplacian function."""

    # Patch cv2.Laplacian to return a constant array
    class DummyCV2:
        CV_64F = 0

        @staticmethod
        def Laplacian(image, flag):
            return np.ones((10, 10))

    monkeypatch.setattr(image_processing, "cv2", DummyCV2)
    image = np.ones((10, 10), dtype=np.uint8)
    var = image_processing.variance_of_laplacian(image)
    assert var == 0.0  # variance of all ones is 0


def test_get_gbif_image_success(monkeypatch):
    """Test fetching an image from GBIF for a valid species."""

    # Patch requests.get for species and occurrence
    class DummyResp:
        def __init__(self, json_data=None, content=None):
            self._json = json_data
            self.content = content

        def json(self):
            return self._json

    # First call: species match
    monkeypatch.setattr(image_processing, "requests", mock.Mock())
    image_processing.requests.get = mock.Mock(
        side_effect=[
            DummyResp(json_data={"usageKey": 123}),
            DummyResp(
                json_data={"results": [{"media": [{"identifier": "http://img.url"}]}]}
            ),
            DummyResp(content=Image.new("RGB", (10, 10)).tobytes()),
        ]
    )
    # Patch Image.open and BytesIO
    monkeypatch.setattr(image_processing, "BytesIO", lambda b: io.BytesIO(b))
    monkeypatch.setattr(image_processing, "Image", Image)
    # Patch Image.open to return an image
    monkeypatch.setattr(Image, "open", lambda f: Image.new("RGB", (10, 10)))
    img = image_processing.get_gbif_image("Panthera leo")
    assert isinstance(img, Image.Image)


def test_get_gbif_image_not_found(monkeypatch):
    """Test fetching an image from GBIF for a species that does not exist."""
    monkeypatch.setattr(image_processing, "requests", mock.Mock())
    image_processing.requests.get = mock.Mock(
        return_value=type("Resp", (), {"json": lambda self: {}})()
    )
    img = image_processing.get_gbif_image("notaspecies")
    assert img is None


def test_get_gbif_image_no_results(monkeypatch):
    monkeypatch.setattr(image_processing, "requests", mock.Mock())
    image_processing.requests.get = mock.Mock(
        side_effect=[
            type("Resp", (), {"json": lambda self: {"usageKey": 123}})(),
            type("Resp", (), {"json": lambda self: {"results": []}})(),
        ]
    )
    img = image_processing.get_gbif_image("Panthera leo")
    assert img is None


def test_get_gbif_image_media_but_load_fails(monkeypatch, capsys):
    # Patch requests.get for species and occurrence
    class DummyResp:
        def __init__(self, json_data=None, content=None):
            self._json = json_data
            self.content = content

        def json(self):
            return self._json

    monkeypatch.setattr(image_processing, "requests", mock.Mock())
    image_processing.requests.get = mock.Mock(
        side_effect=[
            DummyResp(json_data={"usageKey": 123}),
            DummyResp(
                json_data={"results": [{"media": [{"identifier": "http://img.url"}]}]}
            ),
            DummyResp(content=b"fakebytes"),
        ]
    )
    # Patch Image.open to raise an exception
    monkeypatch.setattr(image_processing, "BytesIO", lambda b: io.BytesIO(b))
    monkeypatch.setattr(image_processing, "Image", Image)

    def raise_error(f):
        raise Exception("fail to load")

    monkeypatch.setattr(Image, "open", raise_error)
    img = image_processing.get_gbif_image("Panthera leo")
    assert img is None
    assert "Error loading image for Panthera leo" in capsys.readouterr().out


def test_image_annotation_img_none_reads_file(monkeypatch, tmp_path):
    from PIL import Image
    import amber_inferences.utils.image_processing as image_processing

    # Create a blank image and save to disk
    img = Image.new("RGB", (20, 20), color="white")
    img_path = tmp_path / "test_img.jpg"
    img.save(img_path)
    # Patch plt.imshow to avoid display
    monkeypatch.setattr(image_processing, "plt", mock.Mock(imshow=lambda img: None))
    # Patch ImageDraw.Draw to a dummy to check it's called
    called = {}
    orig_draw = image_processing.ImageDraw.Draw

    def fake_draw(img_arg):
        called["called"] = True
        return orig_draw(img_arg)

    monkeypatch.setattr(image_processing.ImageDraw, "Draw", fake_draw)
    # Call with img=None, should read from file
    image_processing.image_annotation(str(img_path), img=None, boxes=[])
    assert called["called"]
