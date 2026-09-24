"""Deterministic, weight-free image features shared by every demo component."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

EMBEDDING_DIM = 54
BATCH_SIZE = 30
IMAGE_SIZE = 16
POOL_GRID_SIZE = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into an RGB float32 array scaled to [0, 1]."""
    if not isinstance(image_bytes, bytes | bytearray | memoryview):
        raise TypeError("image_bytes must be bytes-like")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"image_bytes exceeds {MAX_IMAGE_BYTES} bytes")

    try:
        with Image.open(io.BytesIO(bytes(image_bytes))) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
            oriented = ImageOps.exif_transpose(image)
            rgb = oriented.convert("RGB")
            resized = rgb.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BILINEAR,
            )
            pixels = np.asarray(resized, dtype=np.float32).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError("unable to decode image bytes") from exc

    if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"decoded image has unexpected shape: {pixels.shape}")

    pixels /= np.float32(255.0)
    if not np.isfinite(pixels).all():
        raise ValueError("decoded image contains non-finite values")
    return pixels


def _embed_pixels(pixels: np.ndarray) -> np.ndarray:
    """Create the 54-value representation from a resized RGB image."""
    pool_factor = IMAGE_SIZE // POOL_GRID_SIZE
    pooled = pixels.reshape(
        POOL_GRID_SIZE,
        pool_factor,
        POOL_GRID_SIZE,
        pool_factor,
        3,
    ).mean(axis=(1, 3), dtype=np.float32)
    # Keep the grid channel-major so the feature order is explicit and stable.
    grid_values = np.transpose(pooled, (2, 0, 1)).reshape(-1)
    channel_mean = pixels.mean(axis=(0, 1), dtype=np.float32)
    channel_std = pixels.std(axis=(0, 1), dtype=np.float32)
    embedding = np.concatenate((grid_values, channel_mean, channel_std)).astype(
        np.float32,
        copy=False,
    )
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(f"encoder produced unexpected shape: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("encoder produced non-finite values")
    return embedding


def embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Encode one image byte string as a finite float32 vector of length 54."""
    return _embed_pixels(_decode_image(image_bytes))


def embed_image_path(path: Path) -> np.ndarray:
    """Read and encode one image file."""
    return embed_image_bytes(path.read_bytes())


def embed_image_batch(image_bytes: Sequence[bytes]) -> np.ndarray:
    """Encode a non-empty sequence of images as an (N, 54) array."""
    if not image_bytes:
        raise ValueError("image batch must not be empty")
    batch = np.stack([embed_image_bytes(item) for item in image_bytes], axis=0)
    if batch.ndim != 2 or batch.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"encoder produced unexpected batch shape: {batch.shape}")
    if not np.isfinite(batch).all():
        raise ValueError("encoder produced non-finite batch values")
    return batch.astype(np.float32, copy=False)


def embedding_columns() -> list[str]:
    """Return the stable names for the 54 embedding dimensions."""
    return [f"embedding-{index}" for index in range(EMBEDDING_DIM)]
