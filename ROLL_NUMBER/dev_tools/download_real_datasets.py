"""
Download and Prepare Datasets from Roboflow and Google Drive
=============================================================
Downloads:
1. Traffic/Motorcycle Dataset (Roboflow)
2. Helmet & License Plate Dataset (Roboflow)
3. License Plate Character Dataset (Google Drive)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request

def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")
    packages = ["roboflow", "gdown"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} already installed")
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"✓ {pkg} installed")


def download_roboflow_dataset(project_url, dataset_name, output_dir):
    """Download dataset from Roboflow"""
    print(f"\n📥 Downloading {dataset_name} from Roboflow...")
    
    try:
        from roboflow import Roboflow
        
        # Extract project info from URL
        # URL format: https://universe.roboflow.com/[workspace]/[project]
        parts = project_url.strip("/").split("/")
        workspace = parts[-2]
        project = parts[-1]
        
        print(f"   Workspace: {workspace}")
        print(f"   Project: {project}")
        
        # Initialize Roboflow (uses API key if available, else downloads public)
        rf = Roboflow(api_key="")  # Leave empty for public datasets
        
        try:
            workspace_obj = rf.workspace(workspace)
            project_obj = workspace_obj.project(project)
            
            # Download dataset
            dataset = project_obj.version(1).download("yolov8", location=output_dir, overwrite=True)
            print(f"✓ {dataset_name} downloaded successfully!")
            return dataset.location
        except Exception as e:
            print(f"⚠️  Could not download from Roboflow API: {e}")
            print(f"   Trying alternative method...")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_google_drive_dataset(drive_url, output_dir):
    """Download dataset from Google Drive"""
    print(f"\n📥 Downloading License Plate Character Dataset from Google Drive...")
    
    try:
        import gdown
        
        # Extract folder ID from URL
        # URL format: https://drive.google.com/drive/folders/[FOLDER_ID]
        folder_id = drive_url.split("/")[-1].split("?")[0]
        
        print(f"   Folder ID: {folder_id}")
        print(f"   Downloading to: {output_dir}")
        
        output_path = os.path.join(output_dir, "lp_character")
        os.makedirs(output_path, exist_ok=True)
        
        # Download folder
        gdown.download_folder(
            url=drive_url,
            output=output_path,
            quiet=False,
            use_cookies=False
        )
        
        print(f"✓ License Plate Character Dataset downloaded successfully!")
        return output_path
        
    except Exception as e:
        print(f"⚠️  Warning: Could not download from Google Drive: {e}")
        print(f"   Manual download needed: {drive_url}")
        return None


def merge_datasets(dataset_paths, output_dir):
    """Merge multiple datasets into single YOLO format"""
    print(f"\n🔄 Merging datasets...")
    
    merged_dir = os.path.join(output_dir, "merged_dataset")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Create YOLO structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(merged_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, "labels", split), exist_ok=True)
    
    total_images = 0
    
    # Copy images and labels from each dataset
    for dataset_path in dataset_paths:
        if not dataset_path or not os.path.exists(dataset_path):
            print(f"⚠️  Skipping {dataset_path} (not found)")
            continue
            
        print(f"   Processing: {dataset_path}")
        
        # Check if this is a YOLO format dataset
        for split in ["train", "val", "test"]:
            src_img_dir = os.path.join(dataset_path, "images", split)
            src_lbl_dir = os.path.join(dataset_path, "labels", split)
            
            if os.path.exists(src_img_dir):
                dst_img_dir = os.path.join(merged_dir, "images", split)
                for img_file in os.listdir(src_img_dir):
                    src = os.path.join(src_img_dir, img_file)
                    dst = os.path.join(dst_img_dir, f"{Path(dataset_path).name}_{img_file}")
                    shutil.copy2(src, dst)
                    total_images += 1
            
            if os.path.exists(src_lbl_dir):
                dst_lbl_dir = os.path.join(merged_dir, "labels", split)
                for lbl_file in os.listdir(src_lbl_dir):
                    src = os.path.join(src_lbl_dir, lbl_file)
                    dst = os.path.join(dst_lbl_dir, f"{Path(dataset_path).name}_{lbl_file}")
                    shutil.copy2(src, dst)
    
    print(f"✓ Merged {total_images} images")
    return merged_dir


def create_data_yaml(dataset_dir, output_file):
    """Create data.yaml for YOLO training"""
    print(f"\n📝 Creating data.yaml...")
    
    yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: motorcycle
  1: rider
  2: helmet
  3: license_plate
"""
    
    with open(output_file, "w") as f:
        f.write(yaml_content)
    
    print(f"✓ Created {output_file}")
    return output_file


def main():
    print("=" * 70)
    print("DATASET DOWNLOAD AND PREPARATION")
    print("=" * 70)
    
    # Install packages
    install_packages()
    
    # Create output directory
    output_dir = "./datasets_real"
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_datasets = []
    
    # 1. Download Motorcycle Detection Dataset
    print("\n" + "=" * 70)
    print("DATASET 1: Traffic/Motorcycle Detection")
    print("=" * 70)
    motorcycle_url = "https://universe.roboflow.com/cdio-zmfmj/motobike-detection"
    motorcycle_dataset = download_roboflow_dataset(
        motorcycle_url,
        "Motorcycle Detection",
        output_dir
    )
    if motorcycle_dataset:
        downloaded_datasets.append(motorcycle_dataset)
    else:
        print("⚠️  Please download manually from:")
        print(f"   {motorcycle_url}")
        print("   Save to: datasets_real/motorcycle_detection")
    
    # 2. Download Helmet & License Plate Dataset
    print("\n" + "=" * 70)
    print("DATASET 2: Helmet & License Plate Detection")
    print("=" * 70)
    helmet_url = "https://universe.roboflow.com/cdio-zmfmj/helmet-lincense-plate-detection-gevlq"
    helmet_dataset = download_roboflow_dataset(
        helmet_url,
        "Helmet & License Plate",
        output_dir
    )
    if helmet_dataset:
        downloaded_datasets.append(helmet_dataset)
    else:
        print("⚠️  Please download manually from:")
        print(f"   {helmet_url}")
        print("   Save to: datasets_real/helmet_lp_detection")
    
    # 3. Download License Plate Character Dataset
    print("\n" + "=" * 70)
    print("DATASET 3: License Plate Character Dataset")
    print("=" * 70)
    lp_drive_url = "https://drive.google.com/drive/folders/1hxB8147kZUgVipVys8VRkcYkOFwAozTD?usp=drive_link"
    lp_dataset = download_google_drive_dataset(
        lp_drive_url,
        output_dir
    )
    if lp_dataset:
        downloaded_datasets.append(lp_dataset)
    else:
        print("⚠️  Please download manually from:")
        print(f"   {lp_drive_url}")
        print("   Save to: datasets_real/lp_character")
    
    # Merge datasets
    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)
    
    if downloaded_datasets:
        merged_dataset = merge_datasets(downloaded_datasets, output_dir)
        
        # Create data.yaml
        data_yaml = os.path.join(merged_dataset, "data.yaml")
        create_data_yaml(merged_dataset, data_yaml)
        
        print("\n" + "=" * 70)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Merged dataset location: {merged_dataset}")
        print(f"📋 YAML config file: {data_yaml}")
        
        print("\n🚀 Next steps to train:")
        print(f"\n   python train_quick.py --stage motorcycle \\")
        print(f"       --data {data_yaml} \\")
        print(f"       --epochs 50 --batch 8")
        
        print("\n" + "=" * 70)
    else:
        print("\n⚠️  No datasets downloaded automatically")
        print("Please download manually from the URLs above")
        print("Then organize in datasets_real/ folder:")
        print("  datasets_real/")
        print("    ├── motorcycle_detection/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    ├── helmet_lp_detection/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    └── lp_character/")


if __name__ == "__main__":
    main()
