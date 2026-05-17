import os
import shutil
import zipfile
import logging
import cv2
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Consolidator")

def clear_dir(path: Path):
    if path.exists():
        logger.info(f"Removing existing directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def process_zip_dataset(zip_path: Path, ds_prefix: str, target_classes: list, class_mapping: dict):
    logger.info(f"Processing {zip_path.name} directly from archive in-memory...")
    
    results = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with zipfile.ZipFile(zip_path, 'r') as zh:
        namelist = zh.namelist()
        namelist_set = set(namelist)
        
        # Filter image files under images/ directories
        image_files = []
        for name in namelist:
            p = Path(name)
            if p.suffix.lower() in image_extensions and "images" in p.parts:
                image_files.append(name)
        
        logger.info(f"Found {len(image_files)} image files in ZIP archive.")
        
        for img_zip_path_str in image_files:
            # Determine parallel label path
            img_parts = list(Path(img_zip_path_str).parts)
            try:
                idx = img_parts.index("images")
                img_parts[idx] = "labels"
                lbl_parts = list(img_parts)
                lbl_parts[-1] = Path(lbl_parts[-1]).stem + ".txt"
                lbl_zip_path_str = "/".join(lbl_parts)
            except ValueError:
                continue
                
            if lbl_zip_path_str not in namelist_set:
                continue
                
            # Read and filter annotations in-memory
            try:
                lbl_content = zh.read(lbl_zip_path_str).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to read label {lbl_zip_path_str} from zip: {e}")
                continue
                
            valid_lines = []
            for line in lbl_content.strip().split("\n"):
                line_parts = line.strip().split()
                if len(line_parts) >= 5:
                    try:
                        class_id = int(line_parts[0])
                        if class_id in target_classes:
                            mapped_id = class_mapping[class_id]
                            valid_lines.append(f"{mapped_id} " + " ".join(line_parts[1:5]) + "\n")
                    except ValueError:
                        pass
                        
            # If we have valid 2-wheeler boxes, process image and append!
            if valid_lines:
                try:
                    # 1. Read image bytes in-memory and decode
                    img_bytes = zh.read(img_zip_path_str)
                    img_np = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                        
                    # 2. Resize image if maximum dimension > 640
                    h, w = img.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > 640:
                        scale = 640.0 / max_dim
                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    
                    # 3. Encode to JPEG with quality 85 to save space
                    success, img_encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not success:
                        continue
                        
                    img_bytes_out = img_encoded.tobytes()
                    results.append((ds_prefix, img_bytes_out, valid_lines))
                except Exception as e:
                    logger.warning(f"Error processing {img_zip_path_str}: {e}")
                    
    logger.info(f"✓ {ds_prefix} completed: {len(results)} valid 2-wheeler images.")
    return results

def main():
    root_dir = Path("c:/Users/Lenovo/Documents/cv_project")
    bike_data_dir = root_dir / "bike_data_last"
    
    # 1. Dataset configurations
    zips = {
        "ds1": ("Indian_Vehicle_Detection.v1-cv.yolov11.zip", [1], {1: 0}), # Class 1 (Bike) -> 0
        "ds2": ("Vehicle Detection.v1-cv.yolov11.zip", [2], {2: 0}),       # Class 2 (motorbike) -> 0
        "ds3": ("two-wheeler.v1-cv.yolov11.zip", [0, 1], {0: 0, 1: 0})     # Class 0 (bike) & Class 1 (scooter) -> 0
    }
    
    all_records = []
    
    for ds_prefix, (zip_name, target_classes, mapping) in zips.items():
        zip_path = bike_data_dir / zip_name
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            continue
        
        records = process_zip_dataset(zip_path, ds_prefix, target_classes, mapping)
        all_records.extend(records)
        
    total_imgs = len(all_records)
    total_boxes = sum(len(lines) for _, _, lines in all_records)
    
    logger.info(f"\n========================================")
    logger.info(f"CONSOLIDATION COMPLETE!")
    logger.info(f"Total Unique 2-Wheeler Images: {total_imgs}")
    logger.info(f"Total 2-Wheeler Bounding Boxes: {total_boxes}")
    logger.info(f"========================================\n")
    
    # 2. Distribute records into 5 balanced chunks (interleaved)
    logger.info("Distributing dataset into 5 equal parts...")
    num_parts = 5
    parts_data = [[] for _ in range(num_parts)]
    for idx, record in enumerate(all_records):
        parts_data[idx % num_parts].append(record)
        
    # 3. Create a ZIP for each part
    output_dir = root_dir / "training_lab" / "hf_cv_space" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean any existing MASTER_BIKE zips to prevent mismatch
    for zip_file in output_dir.glob("MASTER_BIKE*.zip"):
        logger.info(f"Removing old zip file: {zip_file}")
        zip_file.unlink()
        
    for p_idx in range(num_parts):
        part_num = p_idx + 1
        part_zip_path = output_dir / f"MASTER_BIKE_PART_{part_num}.zip"
        
        logger.info(f"Building {part_zip_path.name} with {len(parts_data[p_idx])} images...")
        
        # We will build the directory structure in a temporary folder first
        temp_part_dir = root_dir / f"temp_part_{part_num}"
        clear_dir(temp_part_dir)
        
        # Dual paths for the zip
        flat_img = temp_part_dir / "images"
        flat_lbl = temp_part_dir / "labels"
        flat_img.mkdir(exist_ok=True)
        flat_lbl.mkdir(exist_ok=True)
        
        nested_img = temp_part_dir / "MASTER_BIKE" / "images"
        nested_lbl = temp_part_dir / "MASTER_BIKE" / "labels"
        nested_img.mkdir(parents=True, exist_ok=True)
        nested_lbl.mkdir(parents=True, exist_ok=True)
        
        for record_idx, (ds_prefix, img_bytes, valid_lines) in enumerate(parts_data[p_idx]):
            seq_num = record_idx + 1
            new_img_name = f"{ds_prefix}_part{part_num}_{seq_num:06d}.jpg"
            new_lbl_name = f"{ds_prefix}_part{part_num}_{seq_num:06d}.txt"
            
            # Write to flat path
            with open(flat_img / new_img_name, 'wb') as f:
                f.write(img_bytes)
            with open(flat_lbl / new_lbl_name, 'w') as f:
                f.writelines(valid_lines)
                
            # Write to nested path
            with open(nested_img / new_img_name, 'wb') as f:
                f.write(img_bytes)
            with open(nested_lbl / new_lbl_name, 'w') as f:
                f.writelines(valid_lines)
                
        # Zip the temporary part directory
        with zipfile.ZipFile(part_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_h:
            for root, dirs, files in os.walk(temp_part_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(temp_part_dir)
                    zip_h.write(file_path, rel_path)
                    
        logger.info(f"✓ {part_zip_path.name} created successfully!")
        
        # Clean up temporary part directory
        clear_dir(temp_part_dir)
        temp_part_dir.rmdir()
        
    logger.info("\n✓ All 5 parts created successfully under training_lab/hf_cv_space/data/!")

if __name__ == "__main__":
    main()
