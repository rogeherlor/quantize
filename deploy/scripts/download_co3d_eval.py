#!/usr/bin/env python3
"""
Download only the CO3D test-split images needed for VGGT evaluation.

Instead of the full 5.5 TB dataset, this script downloads ~37 GB covering
the 41 SEEN_CATEGORIES test sequences used by run_evaluation_vggt().

Usage:
    python deploy/scripts/download_co3d_eval.py \\
        --output_dir /data/co3d/dataset \\
        --anno_dir   /data/co3d/preprocessed_dataset \\
        [--categories all | apple chair car ...] \\
        [--workers 4] \\
        [--keep_zips] \\
        [--skip_existing]

After completion the directory layout matches what qvggt.py expects:
    output_dir/   <── args.co3d_dir
        {category}/{sequence}/images/*.jpg
    anno_dir/     <── args.co3d_anno_dir
        {category}_test.jgz
        {category}_train.jgz
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Categories used by run_evaluation_vggt
# ---------------------------------------------------------------------------
SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "bottle", "bowl", "broccoli",
    "cake", "car", "carrot", "cellphone", "chair",
    "cup", "donut", "hairdryer", "handbag", "hydrant",
    "keyboard", "laptop", "microwave", "motorcycle", "mouse",
    "orange", "parkingmeter", "pizza", "plant", "stopsign",
    "teddybear", "toaster", "toilet", "toybus", "toyplane",
    "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]

CO3D_LINKS_URL = (
    "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/links.json"
)

MIN_QUALITY = 0.5          # mirrors preprocess_co3d.py default
CHUNK_SIZE  = 1 << 20      # 1 MB download chunks


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_links(cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"[links] using cached {cache_path}")
        with open(cache_path) as f:
            return json.load(f)
    print(f"[links] fetching {CO3D_LINKS_URL} …")
    r = requests.get(CO3D_LINKS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.write_text(json.dumps(data))
    return data


def _download_zip(url: str, dest: Path, desc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc, leave=False
    ) as bar:
        for chunk in r.iter_content(CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------
# Selective ZIP extraction
# ---------------------------------------------------------------------------

def _extract_metadata(zf: zipfile.ZipFile, category: str, output_dir: Path) -> bool:
    """Extract set_lists + frame/sequence annotations if present. Returns True on success."""
    wanted = {
        f"{category}/set_lists/set_lists_fewview_dev.json",
        f"{category}/frame_annotations.jgz",
        f"{category}/sequence_annotations.jgz",
    }
    names_in_zip = set(zf.namelist())
    if not wanted.issubset(names_in_zip):
        return False
    for name in wanted:
        dest = output_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    return True


def _extract_test_images(
    zf: zipfile.ZipFile,
    category: str,
    test_sequences: set,
    output_dir: Path,
    skip_existing: bool,
) -> int:
    """Extract images/  for every entry in test_sequences. Returns image count."""
    prefix   = f"{category}/"
    img_pfx  = "/images/"
    extracted = 0
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        # name: category/seq_name/images/frame000001.jpg
        rest = name[len(prefix):]                    # seq_name/images/frame…
        parts = rest.split("/", 2)
        if len(parts) < 3:
            continue
        seq_name, subdir, filename = parts
        if seq_name not in test_sequences:
            continue
        if subdir != "images":
            continue
        dest = output_dir / name
        if skip_existing and dest.exists():
            continue
        # Handle edge case: if dest.parent is a file (from previous failed extraction),
        # remove it and recreate as directory
        if dest.parent.exists() and dest.parent.is_file():
            dest.parent.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
        extracted += 1
    return extracted


# ---------------------------------------------------------------------------
# Preprocessing — mirrors preprocess_co3d.py logic
# ---------------------------------------------------------------------------

def _preprocess_category(
    category: str,
    co3d_dir: Path,
    anno_dir: Path,
    min_quality: float = MIN_QUALITY,
) -> tuple[int, int]:
    """Generate {category}_{train,test}.jgz from downloaded metadata. Returns (train, test) counts."""
    cat_dir = co3d_dir / category

    set_lists_path   = cat_dir / "set_lists" / "set_lists_fewview_dev.json"
    frame_ann_path   = cat_dir / "frame_annotations.jgz"
    seq_ann_path     = cat_dir / "sequence_annotations.jgz"

    for p in (set_lists_path, frame_ann_path, seq_ann_path):
        if not p.exists():
            print(f"  [preprocess] missing {p} — skipping {category}")
            return 0, 0

    with open(set_lists_path) as f:
        subset_lists = json.load(f)

    with gzip.open(seq_ann_path) as f:
        seq_data = json.loads(f.read())

    with gzip.open(frame_ann_path) as f:
        frame_data_raw = json.loads(f.read())

    # Build frame lookup: seq_name → frame_number → frame_data
    frame_lookup: dict[str, dict] = {}
    for fd in frame_data_raw:
        sn = fd["sequence_name"]
        frame_lookup.setdefault(sn, {})[fd["frame_number"]] = fd

    # Quality filter
    good_seqs = {
        sd["sequence_name"]
        for sd in seq_data
        if sd.get("viewpoint_quality_score", 0) > min_quality
    }

    counts = {}
    for split in ("train", "test"):
        cat_split: dict[str, list] = {}
        for seq_name, frame_number, filepath in subset_lists.get(split, []):
            if seq_name not in good_seqs:
                continue
            fd = frame_lookup.get(seq_name, {}).get(frame_number)
            if fd is None:
                continue
            cat_split.setdefault(seq_name, []).append({
                "filepath":        filepath,
                "R":               fd["viewpoint"]["R"],
                "T":               fd["viewpoint"]["T"],
                "focal_length":    fd["viewpoint"]["focal_length"],
                "principal_point": fd["viewpoint"]["principal_point"],
            })
        out = anno_dir / f"{category}_{split}.jgz"
        with gzip.open(out, "w") as f:
            f.write(json.dumps(cat_split).encode())
        counts[split] = sum(len(v) for v in cat_split.values())

    return counts.get("train", 0), counts.get("test", 0)


# ---------------------------------------------------------------------------
# Per-category pipeline
# ---------------------------------------------------------------------------

def _process_category(
    category: str,
    urls: list[str],
    output_dir: Path,
    anno_dir: Path,
    skip_existing: bool,
    keep_zips: bool,
) -> dict:
    cat_dir        = output_dir / category
    meta_ready     = False
    test_seqs: set[str] = set()
    total_images   = 0
    pending_zips: list[Path] = []   # ZIPs downloaded before metadata was found

    set_lists_path = cat_dir / "set_lists" / "set_lists_fewview_dev.json"
    done_marker    = output_dir / f".{category}.done"

    # Check if category was already successfully processed
    if done_marker.exists():
        with open(done_marker) as f:
            result = json.load(f)
        print(f"[{category}] already processed (cached), skipping download")
        return result

    # Also check if it was processed with an older version (check for final annotation files)
    train_anno = anno_dir / f"{category}_train.jgz"
    test_anno = anno_dir / f"{category}_test.jgz"
    if set_lists_path.exists() and train_anno.exists() and test_anno.exists():
        # Category was already fully processed, just create marker and return
        with open(set_lists_path) as f:
            sl = json.load(f)
        test_seqs = {entry[0] for entry in sl.get("test", [])}
        train_n = len(sl.get("train", []))
        test_n = len(sl.get("test", []))
        
        result = {
            "category": category,
            "images": 0,  # We don't have the count from old run, but it doesn't matter for summary
            "test_seqs": len(test_seqs),
            "train_frames": train_n,
            "test_frames": test_n,
            "status": "ok",
        }
        done_marker.write_text(json.dumps(result))
        print(f"[{category}] already fully processed (from previous run), skipping download")
        return result

    # Metadata may already be present from a previous run
    meta_ready = False
    test_seqs: set[str] = set()
    if set_lists_path.exists():
        with open(set_lists_path) as f:
            sl = json.load(f)
        test_seqs  = {entry[0] for entry in sl.get("test", [])}
        meta_ready = True
        print(f"[{category}] metadata already present, {len(test_seqs)} test seqs")

    for url in urls:
        zip_name = url.split("/")[-1]
        zip_path = output_dir / "_zips" / zip_name

        if not zip_path.exists():
            _download_zip(url, zip_path, f"{category} {zip_name}")
        else:
            print(f"[{category}] already downloaded {zip_name}")

        # Validate zip file integrity; re-download if corrupted
        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            print(f"[{category}] {zip_name} is corrupted, re-downloading…")
            zip_path.unlink(missing_ok=True)
            _download_zip(url, zip_path, f"{category} {zip_name}")
            zf = zipfile.ZipFile(zip_path)
        except Exception as e:
            print(f"[{category}] error opening {zip_name}: {e}, skipping")
            continue

        with zf:
            if not meta_ready:
                if _extract_metadata(zf, category, output_dir):
                    meta_ready = True
                    with open(set_lists_path) as f:
                        sl = json.load(f)
                    test_seqs = {entry[0] for entry in sl.get("test", [])}
                    print(f"[{category}] found metadata in {zip_name}, {len(test_seqs)} test seqs")

                    # Back-fill: extract images from ZIPs downloaded before metadata was found
                    for prev_path in pending_zips:
                        try:
                            prev_zf = zipfile.ZipFile(prev_path)
                        except zipfile.BadZipFile:
                            print(f"[{category}] {prev_path.name} is corrupted, skipping")
                            if not keep_zips:
                                prev_path.unlink(missing_ok=True)
                            continue
                        with prev_zf:
                            n = _extract_test_images(
                                prev_zf, category, test_seqs, output_dir, skip_existing
                            )
                            total_images += n
                            print(f"[{category}] {prev_path.name} (back-fill): {n} images")
                        if not keep_zips:
                            prev_path.unlink(missing_ok=True)
                    pending_zips.clear()

                    # Also extract from the current ZIP
                    n = _extract_test_images(zf, category, test_seqs, output_dir, skip_existing)
                    total_images += n
                    print(f"[{category}] {zip_name}: extracted {n} test images")
                else:
                    # Metadata not yet found — keep this ZIP for potential back-fill
                    pending_zips.append(zip_path)
                    continue
            else:
                n = _extract_test_images(zf, category, test_seqs, output_dir, skip_existing)
                total_images += n
                print(f"[{category}] {zip_name}: extracted {n} test images")

        if not keep_zips:
            zip_path.unlink(missing_ok=True)

    # Clean up any pending ZIPs for which metadata was never found
    for p in pending_zips:
        if not keep_zips:
            p.unlink(missing_ok=True)

    if not meta_ready:
        print(f"[{category}] WARNING: never found metadata — category skipped")
        result = {"category": category, "images": 0, "test_seqs": 0, "status": "no_metadata"}
        return result

    # Preprocess annotations
    anno_dir.mkdir(parents=True, exist_ok=True)
    train_n, test_n = _preprocess_category(category, output_dir, anno_dir)
    print(f"[{category}] annotations: {train_n} train frames, {test_n} test frames")

    result = {
        "category":  category,
        "images":    total_images,
        "test_seqs": len(test_seqs),
        "train_frames": train_n,
        "test_frames":  test_n,
        "status": "ok",
    }

    # Write completion marker for resume support
    done_marker.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download CO3D test-split images for VGGT evaluation"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Destination for CO3D images (maps to args.co3d_dir in qvggt.py)"
    )
    parser.add_argument(
        "--anno_dir", required=True,
        help="Destination for preprocessed .jgz annotation files (maps to args.co3d_anno_dir)"
    )
    parser.add_argument(
        "--categories", nargs="+", default=["all"],
        help="Categories to download, or 'all' for all 41 SEEN_CATEGORIES"
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel category workers (default 1 — sequential)")
    parser.add_argument("--keep_zips", action="store_true",
                        help="Keep downloaded ZIP files after extraction")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip images that already exist on disk (default: on)")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    cats = SEEN_CATEGORIES if args.categories == ["all"] else args.categories
    unknown = set(cats) - set(SEEN_CATEGORIES)
    if unknown:
        print(f"WARNING: unknown categories (will be skipped): {unknown}")
        cats = [c for c in cats if c in SEEN_CATEGORIES]

    output_dir = Path(args.output_dir)
    anno_dir   = Path(args.anno_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    # Fetch CO3D download manifest
    links_cache = output_dir / "_co3d_links.json"
    links = _fetch_links(links_cache)
    full_links: dict[str, list[str]] = links.get("full", links)  # handle both formats

    missing = [c for c in cats if c not in full_links]
    if missing:
        print(f"ERROR: categories not found in links.json: {missing}")
        sys.exit(1)

    print(f"\nDownloading {len(cats)} categories: {', '.join(cats)}\n")

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _process_category,
                    cat, full_links[cat], output_dir, anno_dir,
                    args.skip_existing, args.keep_zips,
                ): cat
                for cat in cats
            }
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for cat in cats:
            results.append(
                _process_category(
                    cat, full_links[cat], output_dir, anno_dir,
                    args.skip_existing, args.keep_zips,
                )
            )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Category':<20} {'Seqs':>6} {'TestFrames':>11} {'Images':>8} {'Status'}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["category"]):
        print(
            f"{r['category']:<20} {r['test_seqs']:>6} "
            f"{r.get('test_frames', 0):>11} {r['images']:>8}  {r['status']}"
        )
    ok  = sum(1 for r in results if r["status"] == "ok")
    bad = len(results) - ok
    print("=" * 60)
    print(f"Done: {ok} categories OK, {bad} failed")
    print(f"Images: {output_dir}")
    print(f"Annotations: {anno_dir}")


if __name__ == "__main__":
    main()
