import pytest
from PIL import Image

import amber_inferences.utils.plotting as plotting


def test_image_annotation_draw(monkeypatch, tmp_path):
    # Create a blank image
    img = Image.new("RGB", (100, 100), color="white")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    boxes = [
        {
            "x_min": 10,
            "y_min": 10,
            "x_max": 50,
            "y_max": 50,
            "label": "A",
            "ann_col": "red",
        },
        {"x_min": 60, "y_min": 60, "x_max": 90, "y_max": 90},
    ]
    # Patch font loading to always use default
    monkeypatch.setattr(
        plotting.ImageFont,
        "truetype",
        lambda *a, **k: plotting.ImageFont.load_default(),
    )
    out_img = plotting.image_annotation(str(img_path), img=img, boxes=boxes)
    assert isinstance(out_img, Image.Image)


def test_fail_with_invalid_boxes():
    dummy_img = Image.new("RGB", (100, 100))

    with pytest.raises(ValueError, match="boxes must be a list of dictionaries"):
        plotting.image_annotation("dummy_path", img=dummy_img, boxes="not_a_list")

    with pytest.raises(ValueError, match="Each box must be a dictionary"):
        plotting.image_annotation(
            "dummy_path", img=dummy_img, boxes=[{"x_min": 10, "y_min": 10}, "notadict"]
        )


def test_fail_with_missing_keys():
    dummy_img = Image.new("RGB", (100, 100))

    with pytest.raises(KeyError, match="'x_max'"):
        plotting.image_annotation(
            "dummy_path", img=dummy_img, boxes=[{"x_min": 10, "y_min": 10}]
        )


def test_image_annotation_scale(monkeypatch, tmp_path):
    img = Image.new("RGB", (300, 300), color="white")
    img_path = tmp_path / "test2.jpg"
    img.save(img_path)
    boxes = [
        {
            "x_min": 30,
            "y_min": 30,
            "x_max": 60,
            "y_max": 60,
            "label": "B",
            "ann_col": "blue",
        }
    ]
    monkeypatch.setattr(
        plotting.ImageFont,
        "truetype",
        lambda *a, **k: plotting.ImageFont.load_default(),
    )
    out_img = plotting.image_annotation(str(img_path), img=img, boxes=boxes, scale=True)
    assert isinstance(out_img, Image.Image)


def test_image_annotation_pass_path(monkeypatch, tmp_path):
    img = Image.new("RGB", (300, 300), color="white")
    img_path = tmp_path / "test2.jpg"
    img.save(img_path)
    boxes = [
        {
            "x_min": 30,
            "y_min": 30,
            "x_max": 60,
            "y_max": 60,
            "label": "B",
            "ann_col": "blue",
        }
    ]
    monkeypatch.setattr(
        plotting.ImageFont,
        "truetype",
        lambda *a, **k: plotting.ImageFont.load_default(),
    )
    out_img = plotting.image_annotation(
        str(img_path), img=None, boxes=boxes, scale=True
    )
    assert isinstance(out_img, Image.Image)


def test_gif_creater(tmp_path):
    # Create a temp dir with images
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (10, 10), color="white")
        img.save(img_dir / f"img{i}.jpg")
    output_gif = tmp_path / "out.gif"
    plotting.gif_creater(str(img_dir), str(output_gif))
    assert output_gif.exists()


def test_gif_creater_invalid_input(tmp_path):
    # Invalid types
    with pytest.raises(ValueError, match="input_dir must be a string or Path object"):
        plotting.gif_creater(123, "output.gif")

    with pytest.raises(ValueError, match="output_path must be a string or Path object"):
        plotting.gif_creater("input_dir", 123)

    # Non-existent input directory
    non_existent_input = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="is not a valid directory"):
        plotting.gif_creater(non_existent_input, tmp_path / "output.gif")

    # Existing input dir, but output has wrong extension
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="must have a .gif extension"):
        plotting.gif_creater(input_dir, tmp_path / "output.txt")

    # Output directory doesn't exist
    nested_output_path = tmp_path / "nonexistent" / "output.gif"
    with pytest.raises(ValueError, match="Output directory .* does not exist"):
        plotting.gif_creater(input_dir, nested_output_path)


def test_gif_creater_not_directory(tmp_path):
    # Invalid types
    with pytest.raises(ValueError, match="input_dir must be a string or Path object"):
        plotting.gif_creater(123, "output.gif")

    with pytest.raises(ValueError, match="output_path must be a string or Path object"):
        plotting.gif_creater("input_dir", 123)

    # Non-existent input directory
    non_existent_input = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="is not a valid directory"):
        plotting.gif_creater(non_existent_input, tmp_path / "output.gif")

    # Existing input dir, but output has wrong extension
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="must have a .gif extension"):
        plotting.gif_creater(input_dir, tmp_path / "output.txt")

    # Output directory doesn't exist
    nested_output_path = tmp_path / "nonexistent" / "output.gif"
    with pytest.raises(ValueError, match="Output directory .* does not exist"):
        plotting.gif_creater(input_dir, nested_output_path)

    # nested_output_path = tmp_path / "nonexistent.temp" / "output.gif"
    # with pytest.raises(ValueError, match="is not a directory"):
    #     plotting.gif_creater(input_dir, nested_output_path)


def test_get_font_default(monkeypatch):
    # Simulate no DejaVuSans font available, should fall back to default
    monkeypatch.setattr("os.path.exists", lambda path: False)

    class DummyFont:
        pass

    monkeypatch.setattr(
        plotting.ImageFont,
        "truetype",
        lambda *a, **k: (_ for _ in ()).throw(Exception("fail")),
    )
    monkeypatch.setattr(plotting.ImageFont, "load_default", lambda: DummyFont())
    font = plotting._get_font()
    assert isinstance(font, DummyFont)


def test_get_font_dejavu(monkeypatch):
    # Simulate DejaVuSans font available
    monkeypatch.setattr("os.path.exists", lambda path: True)

    class DummyFont:
        pass

    monkeypatch.setattr(plotting.ImageFont, "truetype", lambda path, size: DummyFont())
    font = plotting._get_font()
    assert isinstance(font, DummyFont)
