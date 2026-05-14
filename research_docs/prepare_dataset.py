"""
Prepares helmet_dataset.zip and triple_dataset.zip from raw_datasets.
- Normalizes helmet labels: maps all "with helmet" variants → 0, "without helmet" variants → 1
- Triple-rider: keeps person (class 2) and motorcycle (class 1), remaps cleanly
"""
import os
import shutil
import uuid
import zipfile

RAW = "raw_datasets"

# ── Helmet class mappings ──────────────────────────────────────────────────────
# helmet_detection_1: nc=5, names=['Helmet','With Helmet','Without Helmet','helmet','with-helmet']
#   0=Helmet(with), 1=With Helmet(with), 2=Without Helmet(no), 3=helmet(with), 4=with-helmet(with)
HELMET_MAP_1 = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}

# helmet_detection_2: nc=2, names=['With Helmet','Without Helmet']
#   0=With Helmet(with), 1=Without Helmet(no)
HELMET_MAP_2 = {0: 0, 1: 1}

# ── Triple-rider class mappings ────────────────────────────────────────────────
# triple_sharing: nc=3, names=['Class Name...(junk)', 'motorcycle', 'person']
#   0=junk, 1=motorcycle, 2=person
# We keep: motorcycle→0, person→1 (drop class 0)
TRIPLE_MAP = {1: 0, 2: 1}


def copy_yolo_dataset(src_dir, dest_dir, class_map=None, prefix=""):
    """Copy a YOLO dataset, optionally remapping class IDs."""
    for split in ["train", "valid", "test"]:
        src_img = os.path.join(src_dir, split, "images")
        src_lbl = os.path.join(src_dir, split, "labels")
        dst_img = os.path.join(dest_dir, split, "images")
        dst_lbl = os.path.join(dest_dir, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        if not os.path.exists(src_img):
            print(f"  [SKIP] {split}/images not found in {src_dir}")
            continue

        count = 0
        for img_file in os.listdir(src_img):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(img_file)[0]
            lbl_file = base + ".txt"
            new_base = f"{prefix}{uuid.uuid4().hex[:6]}_{base}"
            new_img = new_base + os.path.splitext(img_file)[1]
            new_lbl = new_base + ".txt"

            shutil.copy2(os.path.join(src_img, img_file), os.path.join(dst_img, new_img))

            src_lbl_path = os.path.join(src_lbl, lbl_file)
            dst_lbl_path = os.path.join(dst_lbl, new_lbl)
            if os.path.exists(src_lbl_path):
                if class_map:
                    remap_labels(src_lbl_path, dst_lbl_path, class_map)
                else:
                    shutil.copy2(src_lbl_path, dst_lbl_path)
                count += 1
        print(f"  [{split}] Copied {count} samples")


def remap_labels(src_path, dst_path, class_map):
    """Read YOLO label file, remap class IDs, skip unmapped classes."""
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls in class_map:
                parts[0] = str(class_map[cls])
                lines_out.append(" ".join(parts))
    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out))


def make_zip(src_dir, zip_name):
    print(f"Zipping {src_dir} -> {zip_name}.zip ...")
    shutil.make_archive(zip_name, "zip", src_dir)
    size_mb = os.path.getsize(zip_name + ".zip") / 1e6
    print(f"Done! {zip_name}.zip = {size_mb:.1f} MB")


def prepare_helmet():
    dest = "raw_datasets/helmet_dataset"
    if os.path.exists(dest):
        shutil.rmtree(dest)

    print("\n=== Preparing Helmet Dataset ===")
    print("Copying helmet_detection_1 (5-class -> 2-class)...")
    copy_yolo_dataset(
        f"{RAW}/helmet_detection_1", dest, HELMET_MAP_1, prefix="h1_"
    )
    print("Copying helmet_detection_2 (2-class -> 2-class)...")
    copy_yolo_dataset(
        f"{RAW}/helmet_detection_2", dest, HELMET_MAP_2, prefix="h2_"
    )

    yaml = """train: train/images
val: valid/images
test: test/images

nc: 2
names: ['with_helmet', 'no_helmet']
"""
    with open(os.path.join(dest, "data.yaml"), "w") as f:
        f.write(yaml)
    print("data.yaml written.")


def prepare_triple():
    dest = "raw_datasets/triple_dataset"
    if os.path.exists(dest):
        shutil.rmtree(dest)

    print("\n=== Preparing Triple-Rider Dataset ===")
    print("Copying triple_sharing (remapping: motorcycle->0, person->1)...")
    copy_yolo_dataset(
        f"{RAW}/triple_sharing", dest, TRIPLE_MAP, prefix="tr_"
    )

    yaml = """train: train/images
val: valid/images
test: test/images

nc: 2
names: ['motorcycle', 'person']
"""
    with open(os.path.join(dest, "data.yaml"), "w") as f:
        f.write(yaml)
    print("data.yaml written.")


if __name__ == "__main__":
    prepare_helmet()
    prepare_triple()

    # Count files
    for ds in ["helmet_dataset", "triple_dataset"]:
        path = f"raw_datasets/{ds}/train/images"
        if os.path.exists(path):
            n = len(os.listdir(path))
            print(f"\n{ds}/train/images: {n} files")

    print("\n=== Making ZIPs ===")
    make_zip("raw_datasets/helmet_dataset", "helmet_dataset")
    make_zip("raw_datasets/triple_dataset", "triple_dataset")
    print("\nAll datasets ready! helmet_dataset.zip and triple_dataset.zip created.")
