#!/usr/bin/env python
"""
Build a YOLO-ready *binary* (good/bad) fruit dataset from:

- Custom folders: {fruit}_{good/bad} under ../data/fruits
- Kaggle dataset: ../data/fruits/kaggle_fruits/train + test

Rules:
- 7 custom fruits (avocado, grapes, lemon, mango, pineapple, strawberry, watermelon)
  → split into train/val/test.
- OOD fruits (peach, tomato)
  → go ONLY into test.
- Kaggle fruits (apples, bananas, oranges)
  → kaggle_fruits/train is split into train+val,
    kaggle_fruits/test goes to test.
- Final structure (binary labels):

  ../data/fruits/fruits_yolo_binary/
      train/
          good/
          bad/
      val/
          good/
          bad/
      test/
          good/
          bad/

At the end, creates ../data/fruits/fruits_yolo_binary.zip
"""

import random
import shutil
import zipfile
from pathlib import Path

# ---- config ----
RANDOM_SEED = 42

# for custom fruits
TRAIN_RATIO_CUSTOM = 0.8
VAL_RATIO_CUSTOM = 0.1
TEST_RATIO_CUSTOM = 0.1

# for kaggle_fruits/train → train + val
TRAIN_RATIO_KAGGLE_TRAIN = 0.9  # remaining 0.1 goes to val

CUSTOM_FRUITS = ("avocado", "grapes", "lemon", "mango",
                 "pineapple", "strawberry", "watermelon")
OOD_FRUITS = ("peach", "tomato")  # test-only
KAGGLE_FRUITS = ("apples", "bananas", "oranges")  # for reference only, not used directly

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def copy_files(file_paths, dst_dir: Path):
    """Copy files to dst_dir, avoiding name collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in file_paths:
        src = Path(src)
        if not src.is_file():
            continue
        name = src.name
        dst = dst_dir / name

        # avoid collisions by adding _1, _2, ...
        if dst.exists():
            stem, ext = src.stem, src.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}_{i}{ext}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.copy2(src, dst)


def split_list(items, train_ratio, val_ratio, test_ratio):
    """Split items into train/val/test according to ratios."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    items = list(items)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # ensure all leftovers go to test
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    assert len(test_items) == n_test
    return train_items, val_items, test_items


def split_train_val(items, train_ratio):
    """Split items into train + val (for kaggle train)."""
    items = list(items)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:]
    return train_items, val_items


def main():
    random.seed(RANDOM_SEED)

    # Adjust this if your script lives elsewhere:
    script_dir = Path(__file__).resolve().parent
    fruits_root = (script_dir / ".." / "data" / "fruits").resolve()
    kaggle_root = fruits_root / "kaggle_fruits"
    out_root = fruits_root / "fruits_binary"  # stays inside fruits folder


    print(f"Input root:  {fruits_root}")
    print(f"Kaggle root: {kaggle_root}")
    print(f"Output root: {out_root}")

    # Start fresh
    if out_root.exists():
        print("Removing existing output directory...")
        shutil.rmtree(out_root)

    # Ensure basic dirs exist
    for split in ("train", "val", "test"):
        for label in ("good", "bad"):
            (out_root / split / label).mkdir(parents=True, exist_ok=True)

    # ---- 1) Custom fruits (7 normal) → split into train/val/test ----
    print("\nProcessing custom fruits (train/val/test)...")
    for fruit in CUSTOM_FRUITS:
        for label in ("good", "bad"):
            src_dir = fruits_root / f"{fruit}_{label}"
            if not src_dir.is_dir():
                print(f"  WARNING: missing folder {src_dir}, skipping.")
                continue

            files = [p for p in src_dir.iterdir() if is_image(p)]
            print(f"  {fruit}_{label}: {len(files)} images")

            train_files, val_files, test_files = split_list(
                files,
                TRAIN_RATIO_CUSTOM,
                VAL_RATIO_CUSTOM,
                TEST_RATIO_CUSTOM
            )

            copy_files(train_files, out_root / "train" / label)
            copy_files(val_files, out_root / "val" / label)
            copy_files(test_files, out_root / "test" / label)

    # ---- 2) OOD fruits (peach, tomato) → test-only ----
    print("\nProcessing OOD fruits (test only): peach, tomato...")
    for fruit in OOD_FRUITS:
        for label in ("good", "bad"):
            src_dir = fruits_root / f"{fruit}_{label}"
            if not src_dir.is_dir():
                print(f"  WARNING: missing OOD folder {src_dir}, skipping.")
                continue
            files = [p for p in src_dir.iterdir() if is_image(p)]
            print(f"  {fruit}_{label}: {len(files)} images → test/{label}")
            copy_files(files, out_root / "test" / label)

    # ---- 3) Kaggle fruits ----
    # kaggle_fruits/train → train + val
    # kaggle_fruits/test  → test
    print("\nProcessing Kaggle fruits...")

    kaggle_train_dir = kaggle_root / "train"
    kaggle_test_dir = kaggle_root / "test"

    if not kaggle_train_dir.is_dir() or not kaggle_test_dir.is_dir():
        raise RuntimeError(f"Expected kaggle_fruits/train and kaggle_fruits/test under {kaggle_root}")

    # 3a) kaggle_fruits/train: fresh* / rotten* → train + val
    kaggle_train_good = []
    kaggle_train_bad = []

    print("  Kaggle train → train + val")
    for cls_dir in kaggle_train_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        name = cls_dir.name.lower()
        if name.startswith("fresh"):
            label = "good"
        elif name.startswith("rotten"):
            label = "bad"
        else:
            print(f"    WARNING: unexpected Kaggle train folder name: {name}, skipping.")
            continue

        files = [p for p in cls_dir.iterdir() if is_image(p)]
        print(f"    {name}: {len(files)} images")

        if label == "good":
            kaggle_train_good.extend(files)
        else:
            kaggle_train_bad.extend(files)

    kt_good_train, kt_good_val = split_train_val(kaggle_train_good, TRAIN_RATIO_KAGGLE_TRAIN)
    kt_bad_train, kt_bad_val = split_train_val(kaggle_train_bad, TRAIN_RATIO_KAGGLE_TRAIN)

    copy_files(kt_good_train, out_root / "train" / "good")
    copy_files(kt_bad_train, out_root / "train" / "bad")
    copy_files(kt_good_val, out_root / "val" / "good")
    copy_files(kt_bad_val, out_root / "val" / "bad")

    # 3b) kaggle_fruits/test: fresh* / rotten* → test
    print("\n  Kaggle test → test")
    for cls_dir in kaggle_test_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        name = cls_dir.name.lower()
        if name.startswith("fresh"):
            label = "good"
        elif name.startswith("rotten"):
            label = "bad"
        else:
            print(f"    WARNING: unexpected Kaggle test folder name: {name}, skipping.")
            continue

        files = [p for p in cls_dir.iterdir() if is_image(p)]
        print(f"    {name}: {len(files)} images → test/{label}")
        copy_files(files, out_root / "test" / label)

    # ---- 4) Zip the output, skipping macOS junk ----
    zip_path = fruits_root.parent / "fruits_binary.zip"
    if zip_path.exists():
        zip_path.unlink()

    print(f"\nCreating zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_root.rglob("*"):
            if path.is_dir():
                continue
            # Skip macOS metadata or hidden junk
            if path.name == ".DS_Store" or path.name.startswith("._") or "__MACOSX" in str(path):
                continue
            # Archive path keeps fruits_yolo_binary/ as top-level folder
            arcname = path.relative_to(fruits_root)
            zf.write(path, arcname)

    print("\nDone!")
    print(f"  Final dataset root: {out_root}")
    print(f"  Zip file:          {zip_path}")


if __name__ == "__main__":
    main()
