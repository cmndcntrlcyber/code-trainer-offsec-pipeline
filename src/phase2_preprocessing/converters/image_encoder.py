"""
phase2_preprocessing/converters/image_encoder.py

Encodes PNG screenshots to base64 WebP for HuggingFace dataset storage.
Selects the best representative frame from multi-screenshot captures.
"""
import base64
import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# WebP quality — 85 balances size vs. fidelity for training data
WEBP_QUALITY = 85
# Max dimension for any single axis to keep dataset size manageable
MAX_DIMENSION = 2560


def encode_png_to_base64_webp(
    png_path: Path,
    quality: int = WEBP_QUALITY,
    max_dim: int = MAX_DIMENSION,
) -> str:
    """Convert a single PNG file to base64-encoded WebP bytes."""
    with Image.open(png_path) as img:
        # Downscale if needed while preserving aspect ratio
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="WEBP", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def select_representative_frame(capture_dir: Path) -> Path:
    """
    Choose the best single PNG from a multi-screenshot capture directory.

    Strategy:
    - If only one PNG: return it.
    - If multiple: return the first frame (0000.png), which captures the top
      of the file including imports/class declarations — most useful for training.
    """
    pngs = sorted(capture_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files in {capture_dir}")
    return pngs[0]


def encode_capture_dir(
    capture_dir: Path,
    quality: int = WEBP_QUALITY,
    max_dim: int = MAX_DIMENSION,
) -> str:
    """
    Encode a capture directory's representative screenshot to base64 WebP.

    Args:
        capture_dir: Path to a capture hash directory (containing PNGs + metadata.json)
        quality: WebP quality (1-100)
        max_dim: Max dimension for any axis; image is downscaled with aspect preserved.

    Returns:
        Base64-encoded WebP string
    """
    png = select_representative_frame(capture_dir)
    return encode_png_to_base64_webp(png, quality=quality, max_dim=max_dim)


def encode_all_frames(
    capture_dir: Path,
    quality: int = WEBP_QUALITY,
    max_dim: int = MAX_DIMENSION,
) -> list[str]:
    """
    Encode all PNG frames in a capture directory to base64 WebP.
    Used for multi-frame training samples.

    Returns:
        List of base64-encoded WebP strings, one per frame in sorted order.
    """
    pngs = sorted(capture_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files in {capture_dir}")
    return [encode_png_to_base64_webp(p, quality=quality, max_dim=max_dim) for p in pngs]
