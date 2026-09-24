"""Acquire, verify, and build deterministic KServe image payloads."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image

DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = DEMO_ROOT / "data" / "dataset_manifest.json"
MODEL_NAME = "image-mmd-model"
EXPECTED_CASES = ("identical", "near", "mild", "strong")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CHUNK_SIZE = 1024 * 1024

sys.path.insert(0, str(DEMO_ROOT))

from common.image_features import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    embed_image_batch,
    embed_image_path,
)


class DataPreparationError(ValueError):
    """Raised for invalid manifests, archives, images, or generated payloads."""


class SafeRedirectHandler(HTTPRedirectHandler):
    """Allow archive redirects only when they remain on HTTP(S)."""

    def redirect_request(self, request, file, code, message, headers, new_url):
        parsed_url = urlparse(new_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise DataPreparationError("archive download redirect must use HTTP(S)")
        return super().redirect_request(request, file, code, message, headers, new_url)


DOWNLOAD_OPENER = build_opener(SafeRedirectHandler)


@dataclass(frozen=True)
class ImageSelection:
    """One archive member and its human-readable experimental condition."""

    member: str
    condition: str


@dataclass(frozen=True)
class Sample:
    """One aligned reference row and its current-case image rows."""

    sample_id: str
    reference: ImageSelection
    current: dict[str, ImageSelection]


@dataclass(frozen=True)
class DatasetManifest:
    """Validated manifest values used by all preparation stages."""

    path: Path
    dataset: dict[str, object]
    source: dict[str, str]
    sample_size: int
    cases: dict[str, dict[str, object]]
    samples: tuple[Sample, ...]


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DataPreparationError(f"{description} must be an object")
    return value


def _string(mapping: Mapping[str, object], key: str, description: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DataPreparationError(f"{description}.{key} must be a non-empty string")
    return value


def _safe_archive_member(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataPreparationError(f"{description} must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DataPreparationError(
            f"{description} must be a relative archive member without path traversal"
        )
    return value


def _parse_selection(value: object, description: str) -> ImageSelection:
    selection = _mapping(value, description)
    return ImageSelection(
        member=_safe_archive_member(selection.get("member"), f"{description}.member"),
        condition=_string(selection, "condition", description),
    )


def load_manifest(path: Path) -> DatasetManifest:
    """Load and validate a dataset-neutral manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"unable to read manifest {path}: {exc}") from exc

    document = _mapping(raw, "manifest")
    if document.get("schema_version") != 1:
        raise DataPreparationError("manifest schema_version must be 1")

    dataset = dict(_mapping(document.get("dataset"), "dataset"))
    _string(dataset, "name", "dataset")
    _string(dataset, "homepage", "dataset")

    source_document = _mapping(document.get("source"), "source")
    source_kind = _string(source_document, "kind", "source")
    if source_kind != "zip":
        raise DataPreparationError(
            f"unsupported source kind {source_kind!r}; only 'zip' is implemented"
        )
    source_url = _string(source_document, "url", "source")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise DataPreparationError("source.url must be an absolute HTTP(S) URL")
    source_sha256 = _string(source_document, "sha256", "source").lower()
    if not SHA256_PATTERN.fullmatch(source_sha256):
        raise DataPreparationError("source.sha256 must be a 64-character hex SHA-256")
    filename = source_document.get("filename")
    if filename is None:
        filename = Path(parsed_url.path).name
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise DataPreparationError("source.filename must be a plain archive filename")
    source = {
        "kind": source_kind,
        "url": source_url,
        "sha256": source_sha256,
        "filename": filename,
    }

    sample_size = document.get("sample_size")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise DataPreparationError("sample_size must be an integer")
    if sample_size != BATCH_SIZE:
        raise DataPreparationError(f"sample_size must be {BATCH_SIZE}")

    cases_document = _mapping(document.get("cases"), "cases")
    if tuple(cases_document) != EXPECTED_CASES:
        raise DataPreparationError(
            f"cases must contain {EXPECTED_CASES} in that order"
        )
    cases = {name: dict(_mapping(cases_document[name], f"cases.{name}")) for name in EXPECTED_CASES}

    sample_values = document.get("samples")
    if not isinstance(sample_values, list) or len(sample_values) != sample_size:
        raise DataPreparationError(f"samples must contain exactly {sample_size} rows")

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(sample_values):
        sample_document = _mapping(value, f"samples[{index}]")
        sample_id = _string(sample_document, "id", f"samples[{index}]")
        if sample_id in seen_ids:
            raise DataPreparationError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        reference = _parse_selection(
            sample_document.get("reference"), f"samples[{index}].reference"
        )
        current_document = _mapping(
            sample_document.get("current"), f"samples[{index}].current"
        )
        if tuple(current_document) != EXPECTED_CASES:
            raise DataPreparationError(
                f"samples[{index}].current must contain {EXPECTED_CASES} in that order"
            )
        current = {
            case: _parse_selection(
                current_document[case], f"samples[{index}].current.{case}"
            )
            for case in EXPECTED_CASES
        }
        samples.append(Sample(sample_id, reference, current))

    return DatasetManifest(
        path=path.resolve(),
        dataset=dataset,
        source=source,
        sample_size=sample_size,
        cases=cases,
        samples=tuple(samples),
    )


def selected_members(manifest: DatasetManifest) -> list[str]:
    """Return selected archive members in stable first-seen order."""
    members: list[str] = []
    seen: set[str] = set()
    for sample in manifest.samples:
        selections: Iterable[ImageSelection] = (
            sample.reference,
            *(sample.current[case] for case in EXPECTED_CASES),
        )
        for selection in selections:
            if selection.member not in seen:
                seen.add(selection.member)
                members.append(selection.member)
    return members


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(manifest: DatasetManifest, data_root: Path) -> tuple[Path, Path, Path]:
    raw_root = data_root / "raw"
    archive_path = raw_root / manifest.source["filename"]
    extracted_root = raw_root / "extracted"
    generated_root = data_root / "generated"
    return archive_path, extracted_root, generated_root


def _verify_archive_checksum(manifest: DatasetManifest, archive_path: Path) -> None:
    if not archive_path.is_file():
        raise DataPreparationError(
            f"archive is missing: {archive_path}; run the download command first"
        )
    actual = _sha256_file(archive_path)
    expected = manifest.source["sha256"]
    if actual != expected:
        raise DataPreparationError(
            f"archive checksum mismatch for {archive_path}: expected {expected}, got {actual}"
        )


def _zip_infos(
    archive: zipfile.ZipFile, manifest: DatasetManifest
) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for member in selected_members(manifest):
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise DataPreparationError(
                f"manifest-selected archive member is missing: {member}"
            ) from exc
        if info.is_dir():
            raise DataPreparationError(f"manifest member is a directory: {member}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise DataPreparationError(f"refusing symlink archive member: {member}")
        infos[member] = info
    return infos


def _safe_extraction_target(root: Path, member: str) -> Path:
    target = root / Path(*PurePosixPath(member).parts)
    root_resolved = root.resolve()
    if target.exists() and target.is_symlink():
        raise DataPreparationError(f"refusing to overwrite symlink: {target}")
    if not target.resolve().is_relative_to(root_resolved):
        raise DataPreparationError(f"archive member escapes extraction root: {member}")
    return target


def _extracted_matches(
    archive: zipfile.ZipFile,
    infos: Mapping[str, zipfile.ZipInfo],
    extracted_root: Path,
) -> bool:
    for member, info in infos.items():
        target = _safe_extraction_target(extracted_root, member)
        if not target.is_file() or _sha256_file(target) != _sha256_zip_member(archive, info):
            return False
    return True


def _extract_selected(
    archive: zipfile.ZipFile,
    infos: Mapping[str, zipfile.ZipInfo],
    extracted_root: Path,
) -> None:
    extracted_root.mkdir(parents=True, exist_ok=True)
    for member, info in infos.items():
        target = _safe_extraction_target(extracted_root, member)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=CHUNK_SIZE)


def _download_archive(manifest: DatasetManifest, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{archive_path.name}.",
            suffix=".part",
            dir=archive_path.parent,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            request = Request(
                manifest.source["url"],
                headers={"User-Agent": "odh-trustyai-image-mmd-demo/1"},
            )
            with DOWNLOAD_OPENER.open(request, timeout=60) as response:
                shutil.copyfileobj(response, destination, length=CHUNK_SIZE)
        actual = _sha256_file(temporary_path)
        expected = manifest.source["sha256"]
        if actual != expected:
            raise DataPreparationError(
                f"downloaded archive checksum mismatch: expected {expected}, got {actual}"
            )
        os.replace(temporary_path, archive_path)
        temporary_path = None
    except DataPreparationError:
        raise
    except Exception as exc:
        raise DataPreparationError(f"unable to download {manifest.source['url']}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _ensure_archive_and_extracted(
    manifest: DatasetManifest, data_root: Path
) -> tuple[Path, Path, dict[str, zipfile.ZipInfo]]:
    archive_path, extracted_root, _ = _paths(manifest, data_root)
    if archive_path.exists():
        _verify_archive_checksum(manifest, archive_path)
    else:
        _download_archive(manifest, archive_path)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _zip_infos(archive, manifest)
            if not _extracted_matches(archive, infos, extracted_root):
                _extract_selected(archive, infos, extracted_root)
    except zipfile.BadZipFile as exc:
        raise DataPreparationError(f"archive is not a valid ZIP file: {archive_path}") from exc
    return archive_path, extracted_root, infos


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        embed_image_path(path)
    except (OSError, ValueError) as exc:
        raise DataPreparationError(f"unable to decode selected image {path}: {exc}") from exc


def _verify_source(
    manifest: DatasetManifest, data_root: Path
) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    archive_path, extracted_root, _ = _paths(manifest, data_root)
    _verify_archive_checksum(manifest, archive_path)
    image_hashes: dict[str, dict[str, str]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _zip_infos(archive, manifest)
            for member, info in infos.items():
                target = _safe_extraction_target(extracted_root, member)
                if not target.is_file():
                    raise DataPreparationError(f"selected image was not extracted: {member}")
                _verify_image(target)
                source_hash = _sha256_zip_member(archive, info)
                extracted_hash = _sha256_file(target)
                if source_hash != extracted_hash:
                    raise DataPreparationError(
                        f"extracted image hash mismatch for {member}: "
                        f"expected {source_hash}, got {extracted_hash}"
                    )
                image_hashes[member] = {
                    "archive_sha256": source_hash,
                    "extracted_sha256": extracted_hash,
                }
    except zipfile.BadZipFile as exc:
        raise DataPreparationError(f"archive is not a valid ZIP file: {archive_path}") from exc
    return archive_path, extracted_root, image_hashes


def verify_source(
    manifest: DatasetManifest, data_root: Path
) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    """Verify the pinned archive, selected members, and decoded images."""
    return _verify_source(manifest, data_root)


def _selection_path(extracted_root: Path, selection: ImageSelection) -> Path:
    return _safe_extraction_target(extracted_root, selection.member)


def _base64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(
                document,
                destination,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            destination.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _reference_payload(
    manifest: DatasetManifest,
    extracted_root: Path,
) -> tuple[dict[str, object], list[bytes], np.ndarray]:
    reference_bytes = [
        _selection_path(extracted_root, sample.reference).read_bytes()
        for sample in manifest.samples
    ]
    reference_embeddings = embed_image_batch(reference_bytes)
    payload = {
        "model_name": MODEL_NAME,
        "data_tag": "IMAGE_BASELINE",
        "request": {
            "id": "reference",
            "inputs": [
                {
                    "name": "image",
                    "shape": [manifest.sample_size],
                    "datatype": "BYTES",
                    "data": [base64.b64encode(item).decode("ascii") for item in reference_bytes],
                }
            ],
        },
        "response": {
            "model_name": MODEL_NAME,
            "id": "reference",
            "outputs": [
                {
                    "name": "embedding",
                    "shape": [manifest.sample_size, EMBEDDING_DIM],
                    "datatype": "FP32",
                    "data": reference_embeddings.reshape(-1).astype(float).tolist(),
                }
            ],
        },
    }
    return payload, reference_bytes, reference_embeddings


def _current_payload(
    manifest: DatasetManifest,
    extracted_root: Path,
    case: str,
) -> tuple[dict[str, object], list[bytes]]:
    current_bytes = [
        _selection_path(extracted_root, sample.current[case]).read_bytes()
        for sample in manifest.samples
    ]
    payload = {
        "model_name": MODEL_NAME,
        "id": case,
        "inputs": [
            {
                "name": "image",
                "shape": [manifest.sample_size],
                "datatype": "BYTES",
                "data": [base64.b64encode(item).decode("ascii") for item in current_bytes],
            }
        ],
    }
    return payload, current_bytes


def _encoder_document() -> dict[str, object]:
    return {
        "name": "rgb-average-pool",
        "input_orientation": "EXIF transpose",
        "color_space": "RGB",
        "resize": [16, 16],
        "pool_grid": [4, 4],
        "pool_order": "channel-major",
        "summary": ["rgb_mean", "rgb_population_std"],
        "dimension": EMBEDDING_DIM,
    }


def build_payloads(manifest: DatasetManifest, data_root: Path) -> None:
    """Build the reference upload, current requests, and audit manifest."""
    _, extracted_root, image_hashes = _verify_source(manifest, data_root)
    _, _, generated_root = _paths(manifest, data_root)

    reference_payload, reference_bytes, reference_embeddings = _reference_payload(
        manifest, extracted_root
    )
    current_payloads: dict[str, dict[str, object]] = {}
    current_bytes_by_case: dict[str, list[bytes]] = {}
    for case in EXPECTED_CASES:
        current_payload, current_bytes = _current_payload(manifest, extracted_root, case)
        current_payloads[case] = current_payload
        current_bytes_by_case[case] = current_bytes

    reference_path = generated_root / "reference_upload.json"
    _write_json(reference_path, reference_payload)
    current_paths: dict[str, Path] = {}
    for case, payload in current_payloads.items():
        path = generated_root / f"current_{case}.json"
        _write_json(path, payload)
        current_paths[case] = path

    sample_audit: list[dict[str, object]] = []
    for sample in manifest.samples:
        current_audit = {
            case: {
                "member": sample.current[case].member,
                "condition": sample.current[case].condition,
                "sha256": image_hashes[sample.current[case].member]["extracted_sha256"],
            }
            for case in EXPECTED_CASES
        }
        sample_audit.append(
            {
                "id": sample.sample_id,
                "reference": {
                    "member": sample.reference.member,
                    "condition": sample.reference.condition,
                    "sha256": image_hashes[sample.reference.member]["extracted_sha256"],
                },
                "current": current_audit,
            }
        )

    payload_documents: dict[str, object] = {
        "reference_upload.json": {
            "path": reference_path.name,
            "request_shape": [manifest.sample_size],
            "response_shape": [manifest.sample_size, EMBEDDING_DIM],
            "sha256": _sha256_file(reference_path),
        }
    }
    for case, path in current_paths.items():
        payload_documents[path.name] = {
            "path": path.name,
            "request_shape": [manifest.sample_size],
            "sha256": _sha256_file(path),
        }

    audit = {
        "schema_version": 1,
        "dataset": manifest.dataset,
        "source": manifest.source,
        "sample_size": manifest.sample_size,
        "cases": manifest.cases,
        "encoder": _encoder_document(),
        "samples": sample_audit,
        "payloads": payload_documents,
        "reference_embedding_shape": list(reference_embeddings.shape),
        "reference_image_count": len(reference_bytes),
        "current_image_counts": {
            case: len(values) for case, values in current_bytes_by_case.items()
        },
    }
    _write_json(generated_root / "audit_manifest.json", audit)


def _validate_request(
    payload: object,
    expected_count: int,
    description: str,
) -> list[bytes]:
    document = _mapping(payload, description)
    if document.get("model_name") != MODEL_NAME:
        raise DataPreparationError(f"{description}.model_name must be {MODEL_NAME!r}")
    request = document.get("request", document)
    request_document = _mapping(request, f"{description}.request")
    inputs = request_document.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise DataPreparationError(f"{description} must contain one image input")
    image_input = _mapping(inputs[0], f"{description}.inputs[0]")
    if image_input.get("name") != "image":
        raise DataPreparationError(f"{description} image input name must be 'image'")
    if image_input.get("datatype") != "BYTES":
        raise DataPreparationError(f"{description} image input datatype must be BYTES")
    if image_input.get("shape") != [expected_count]:
        raise DataPreparationError(
            f"{description} image input shape must be [{expected_count}]"
        )
    encoded_values = image_input.get("data")
    if not isinstance(encoded_values, list) or len(encoded_values) != expected_count:
        raise DataPreparationError(
            f"{description} image input must contain {expected_count} values"
        )

    decoded_values: list[bytes] = []
    for index, encoded in enumerate(encoded_values):
        if not isinstance(encoded, str):
            raise DataPreparationError(f"{description} image data {index} is not a string")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise DataPreparationError(
                f"{description} image data {index} is not valid base64"
            ) from exc
        if not decoded:
            raise DataPreparationError(f"{description} image data {index} is empty")
        try:
            with Image.open(io.BytesIO(decoded)) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise DataPreparationError(
                f"{description} image data {index} is not a decodable image"
            ) from exc
        decoded_values.append(decoded)
    return decoded_values


def _validate_response(
    response: object,
    expected_count: int,
    description: str,
) -> None:
    document = _mapping(response, description)
    if document.get("model_name") != MODEL_NAME:
        raise DataPreparationError(f"{description}.model_name must be {MODEL_NAME!r}")
    outputs = document.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise DataPreparationError(f"{description} must contain one output")
    output = _mapping(outputs[0], f"{description}.outputs[0]")
    if output.get("name") != "embedding":
        raise DataPreparationError(f"{description} output name must be 'embedding'")
    if output.get("datatype") != "FP32":
        raise DataPreparationError(f"{description} output datatype must be FP32")
    if output.get("shape") != [expected_count, EMBEDDING_DIM]:
        raise DataPreparationError(
            f"{description} output shape must be [{expected_count},{EMBEDDING_DIM}]"
        )
    values = output.get("data")
    expected_values = expected_count * EMBEDDING_DIM
    if not isinstance(values, list) or len(values) != expected_values:
        raise DataPreparationError(
            f"{description} output must contain {expected_values} values"
        )
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise DataPreparationError(f"{description} output contains non-numeric values")
    if not all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise DataPreparationError(f"{description} output contains non-finite values")


def verify_payloads(manifest: DatasetManifest, data_root: Path) -> None:
    """Verify source files and every generated request/response contract."""
    _verify_source(manifest, data_root)
    _, _, generated_root = _paths(manifest, data_root)
    reference_path = generated_root / "reference_upload.json"
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(
            f"reference payload is missing or invalid: {reference_path}"
        ) from exc
    reference_document = _mapping(reference, "reference_upload")
    if reference_document.get("data_tag") != "IMAGE_BASELINE":
        raise DataPreparationError("reference_upload.data_tag must be IMAGE_BASELINE")
    _validate_request(reference, manifest.sample_size, "reference_upload")
    _validate_response(
        reference_document.get("response"),
        manifest.sample_size,
        "reference_upload.response",
    )

    for case in EXPECTED_CASES:
        path = generated_root / f"current_{case}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataPreparationError(f"current payload is missing or invalid: {path}") from exc
        _validate_request(payload, manifest.sample_size, f"current_{case}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    def add_data_options(target: argparse.ArgumentParser, *, global_options: bool) -> None:
        target.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_MANIFEST if global_options else argparse.SUPPRESS,
        )
        target.add_argument(
            "--data-root",
            type=Path,
            default=None if global_options else argparse.SUPPRESS,
            help="directory containing raw/ and generated/ (defaults beside the manifest)",
        )

    add_data_options(parser, global_options=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("download", "download and extract selected members"),
        ("build", "build generated request/response payloads"),
        ("verify", "verify source files and generated payloads"),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        add_data_options(command_parser, global_options=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else manifest_path.parent
    )
    try:
        manifest = load_manifest(manifest_path)
        if args.command == "download":
            archive_path, extracted_root, infos = _ensure_archive_and_extracted(
                manifest, data_root
            )
            print(
                f"verified {archive_path} and {len(infos)} selected members under {extracted_root}"
            )
        elif args.command == "build":
            build_payloads(manifest, data_root)
            print(f"built generated payloads under {data_root / 'generated'}")
        else:
            verify_payloads(manifest, data_root)
            print(
                f"verified {manifest.sample_size} samples, {len(selected_members(manifest))} "
                f"selected members, and generated payloads"
            )
    except DataPreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
