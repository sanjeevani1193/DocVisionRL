"""Inspect exactly three CORD v2 training rows without materializing a split."""

import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# Set cache locations before importing Hugging Face libraries. Respect user overrides.
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from datasets import Image as DatasetImage, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, UnidentifiedImageError

DATASET = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SAMPLE_COUNT = 3
PREVIEW_CHARS = 1800


def preview(value, limit=PREVIEW_CHARS):
    """Pretty-print JSON, explicitly marking a display-only truncation."""
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def collect_types(value, path, observations):
    """Walk actual keys; [] merges list indices without flattening the data."""
    observations[path][type(value).__name__] += 1
    if isinstance(value, dict):
        for key, child in value.items():
            collect_types(child, f"{path}.{key}", observations)
    elif isinstance(value, list):
        for child in value:
            collect_types(child, f"{path}[]", observations)


def inspect_image(raw):
    """Decode each image separately so a bad image cannot hide its annotation."""
    try:
        if not isinstance(raw, dict):
            raise TypeError(f"expected image storage dict, got {type(raw).__name__}")
        source = io.BytesIO(raw["bytes"]) if raw.get("bytes") is not None else raw.get("path")
        if source is None:
            raise ValueError("image has neither bytes nor path")
        with Image.open(source) as image:
            image.load()  # Verify actual pixels, rather than just reading the header.
            size = image.size
            print(f"Image: storage={type(raw).__name__}, decoded={type(image).__name__}, "
                  f"size={size} (width, height), mode={image.mode}, format={image.format}")
            return size
    except (OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
        print(f"WARNING: image cannot be decoded: {exc}")
        return None


def inspect_annotation(raw, size, observations, presence):
    """Validate the JSON container, report missing data, preserve all source values."""
    print(f"ground_truth storage type: {type(raw).__name__}")
    try:
        annotation = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        print(f"WARNING: invalid ground_truth JSON: {exc}")
        return False
    if not isinstance(annotation, dict):
        print(f"WARNING: ground_truth must be an object; got {type(annotation).__name__}")
        return False

    print("Annotation root fields (type and container length):")
    for key, value in annotation.items():
        length = f", length={len(value)}" if isinstance(value, (dict, list, str)) else ""
        print(f"  {key}: {type(value).__name__}{length}")
    meta = annotation.get("meta")
    print("meta:", preview(meta, 600))
    if isinstance(meta, dict) and size is not None:
        declared = meta.get("image_size")
        if isinstance(declared, dict):
            expected = (declared.get("width"), declared.get("height"))
            if expected != size:
                print(f"WARNING: meta.image_size={expected} differs from decoded size={size}")

    if "gt_parse" not in annotation:
        print("WARNING: missing gt_parse")
        return False
    parsed = annotation["gt_parse"]
    print("gt_parse preview (source values preserved):\n" + preview(parsed))
    row_types = defaultdict(Counter)
    collect_types(parsed, "gt_parse", row_types)
    for path, counts in sorted(row_types.items()):
        print(f"  {path}: {dict(counts)}")
        observations[path].update(counts)
        presence[path] += 1
    if not isinstance(parsed, dict) or not parsed:
        print("WARNING: gt_parse is not a nonempty object")
        return False
    return True


def main():
    print(f"Dataset: {DATASET}\nPinned revision: {REVISION}")
    info = HfApi().dataset_info(DATASET, revision=REVISION)
    print(f"Resolved revision: {info.sha}; last modified: {info.last_modified}")
    if info.sha != REVISION:
        raise RuntimeError("Resolved dataset revision does not match the pinned revision")
    print("Dataset card:", preview(info.card_data.to_dict() if info.card_data else {}))

    # This small metadata file does not contain receipt images or sample rows.
    metadata_path = hf_hub_download(DATASET, "dataset_infos.json", repo_type="dataset", revision=REVISION)
    with open(metadata_path, encoding="utf-8") as handle:
        print("Published dataset metadata:\n" + preview(json.load(handle), 6000))

    # No download_and_prepare(), non-streaming fallback, shuffle, or worker pool.
    stream = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    print(f"Dataset object: {type(stream).__name__}")
    print(f"Available features: {stream.features}")
    # Keep raw image storage so decoding failures can be reported per row.
    stream = stream.cast_column("image", DatasetImage(decode=False))
    observations = defaultdict(Counter)
    presence = Counter()
    count = images_ok = annotations_ok = 0
    for count, example in enumerate(stream.take(SAMPLE_COUNT), start=1):
        print(f"\n--- Training example {count} (zero-based index {count - 1}) ---")
        print("Row keys:", sorted(example))
        size = inspect_image(example.get("image"))
        images_ok += size is not None
        annotations_ok += inspect_annotation(
            example.get("ground_truth"), size, observations, presence
        )

    print("\n--- Observed gt_parse paths across these three rows only ---")
    print("Counts refer to values; row presence counts each path once per receipt.")
    for path, counts in sorted(observations.items()):
        print(f"{path}: types={dict(counts)}, rows={presence[path]}/{count}")
    print(f"\nInspected {count}/{SAMPLE_COUNT} rows; decoded images={images_ok}; "
          f"valid nonempty gt_parse objects={annotations_ok}.")
    print("Streaming may fetch Parquet blocks/row groups beyond three rows; "
          "this is not a byte-level download limit. No complete split is materialized.")
    if (count, images_ok, annotations_ok) != (SAMPLE_COUNT,) * 3:
        raise RuntimeError("Inspection incomplete; see warnings above")


if __name__ == "__main__":
    main()
