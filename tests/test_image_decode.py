"""load_image_from_bytes: orientation, formats, pixel budget, transparency, downscaling."""

import io

import numpy as np
import pytest
from PIL import Image

from face_recognition_service.config import settings
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils.image_utils import (
    ImageProcessingError,
    load_image_from_bytes,
)


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def _two_tone(width: int, height: int) -> Image.Image:
    """Left half red, right half blue."""
    image = Image.new("RGB", (width, height), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, width // 2, height))
    return image


def test_exif_orientation_is_applied() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated; display = rotate 90° clockwise
    data = _encode(_two_tone(80, 40), "JPEG", exif=exif.tobytes(), quality=95)
    bgr = load_image_from_bytes(data)
    assert bgr.shape[:2] == (80, 40)  # height, width swapped
    top, bottom = bgr[10, 20], bgr[70, 20]
    assert top[2] > 200 and top[0] < 60  # red (BGR) on top: the left half rotated to the top
    assert bottom[0] > 200 and bottom[2] < 60


def test_exif_transpose_failure_falls_back_to_undecoded_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_image):
        raise OSError("corrupt EXIF block")

    monkeypatch.setattr("face_recognition_service.utils.image_utils.ImageOps.exif_transpose", _boom)

    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated; would normally swap width/height on display
    data = _encode(_two_tone(80, 40), "JPEG", exif=exif.tobytes(), quality=95)
    bgr = load_image_from_bytes(data)
    # exif_transpose raised, so the image decodes un-rotated: same shape as
    # the un-rotated (width, height) source rather than the swapped shape
    # test_exif_orientation_is_applied asserts for a successful transpose.
    assert bgr.shape[:2] == (40, 80)


def test_mpo_is_accepted() -> None:
    first, second = _two_tone(64, 48), _two_tone(64, 48)
    data = _encode(first, "MPO", save_all=True, append_images=[second])
    assert Image.open(io.BytesIO(data)).format == "MPO"
    assert load_image_from_bytes(data).shape == (48, 64, 3)


def test_transparency_is_composited_onto_white() -> None:
    rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))  # fully transparent black
    bgr = load_image_from_bytes(_encode(rgba, "PNG"))
    assert bgr.min() >= 250


def test_pixel_budget_is_enforced_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_image_pixels", 100)
    with pytest.raises(ImageProcessingError) as exc_info:
        load_image_from_bytes(_encode(Image.new("RGB", (20, 20)), "PNG"))
    assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_large_images_are_downscaled_keeping_aspect(fmt: str) -> None:
    bgr = load_image_from_bytes(_encode(Image.new("RGB", (4096, 1366), (10, 20, 30)), fmt))
    height, width = bgr.shape[:2]
    assert width == 2048
    assert height == pytest.approx(683, abs=2)


def test_downscaling_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_image_side", 0)
    assert load_image_from_bytes(_encode(Image.new("RGB", (3000, 100)), "PNG")).shape[:2] == (100, 3000)


def test_small_images_are_untouched() -> None:
    assert load_image_from_bytes(_encode(Image.new("RGB", (640, 480)), "PNG")).shape == (480, 640, 3)


def test_unsupported_format_and_corrupt_bytes_keep_their_codes() -> None:
    with pytest.raises(ImageProcessingError) as unsupported:
        load_image_from_bytes(_encode(Image.new("RGB", (40, 40)), "TIFF"))
    assert unsupported.value.error_code == ErrorCode.UNSUPPORTED_FORMAT
    with pytest.raises(ImageProcessingError) as corrupt:
        load_image_from_bytes(b"definitely not an image")
    assert corrupt.value.error_code == ErrorCode.INVALID_IMAGE


def test_grayscale_becomes_three_channel() -> None:
    assert load_image_from_bytes(_encode(Image.new("L", (50, 40), 128), "PNG")).shape == (40, 50, 3)


def test_output_is_bgr_uint8() -> None:
    bgr = load_image_from_bytes(_encode(Image.new("RGB", (32, 32), (255, 0, 0)), "PNG"))
    assert bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [0, 0, 255]
