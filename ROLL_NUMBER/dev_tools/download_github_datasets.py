"""
Download Datasets from GitHub Repositories for Training
Fetches pre-collected, annotated datasets from leading traffic violation detection projects

Sources:
  1. kashishparmar02/triple-rider-detection - 6000+ annotated images
  2. RonLek/ALPR-and-Identification-for-Indian-Vehicles - 2500+ difficult cases
  3. ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16 - Helmet annotations
  4. Public datasets (CCPD, RideSafe references)
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from urllib.parse import urljoin
import subprocess

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Install required packages: pip install requests tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubDatasetDownloader:
    """Download datasets from GitHub repositories."""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                        for chunk in response.iter_content(chunk_size):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size):
                        f.write(chunk)
            
            logger.info(f"✓ Downloaded: {dest_path.name}")
            return True
        
        except Exception as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            return False
    
    def download_github_repo_file(self, owner: str, repo: str, path: str, 
                                 dest: Path, branch: str = "main") -> bool:
        """Download file from GitHub repository."""
        # GitHub raw content URL
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        logger.info(f"Downloading from: {url}")
        return self.download_file(url, dest)
    
    def download_kashishparmar02_dataset(self) -> bool:
        """
        Download triple-rider detection dataset (6000+ images).
        
        Dataset contains:
        - Motorcycle images with rider annotations
        - Triple-riding examples
        - Various angles and lighting conditions
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 1: KashishParmar02 - Triple Rider Detection (6000+ images)")
        logger.info("="*70)
        
        try:
            # Clone repository (if dataset is in repo)
            repo_url = "https://github.com/kashishparmar02/triple-rider-detection"
            dest_path = self.output_dir / "kashishparmar02_triple_rider"
            
            if dest_path.exists():
                logger.info(f"Dataset already exists at {dest_path}")
                return True
            
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dest_path)],
                check=True,
                capture_output=True
            )
            
            # Look for datasets folder
            datasets_folder = dest_path / "datasets"
            if datasets_folder.exists():
                logger.info(f"✓ Found datasets folder: {datasets_folder}")
                img_count = sum(1 for f in datasets_folder.rglob("*.jpg"))
                img_count += sum(1 for f in datasets_folder.rglob("*.png"))
                logger.info(f"✓ Total images: {img_count}")
                return True
            else:
                logger.warning("Datasets folder not found in repository")
                # Check for images in root
                img_count = sum(1 for f in dest_path.rglob("*.jpg"))
                if img_count > 0:
                    logger.info(f"✓ Found {img_count} images in repository")
                    return True
                return False
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to clone repository: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error downloading dataset: {e}")
            return False
    
    def download_ronlek_dataset(self) -> bool:
        """
        Download ALPR Indian Vehicles dataset (2500+ difficult cases).
        
        Dataset contains:
        - License plate images with OCR labels
        - Low-light images
        - Motion blur examples
        - Occlusion cases
        - Indian HSRP formats
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 2: RonLek - ALPR Indian Vehicles (2500+ difficult cases)")
        logger.info("="*70)
        
        try:
            repo_url = "https://github.com/RonLek/ALPR-and-Identification-for-Indian-Vehicles"
            dest_path = self.output_dir / "ronlek_alpr_indian"
            
            if dest_path.exists():
                logger.info(f"Dataset already exists at {dest_path}")
                return True
            
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dest_path)],
                check=True,
                capture_output=True
            )
            
            # Look for dataset files
            data_folders = ["data", "dataset", "datasets", "ALPR"]
            for folder_name in data_folders:
                folder_path = dest_path / folder_name
                if folder_path.exists():
                    img_count = sum(1 for f in folder_path.rglob("*.jpg"))
                    img_count += sum(1 for f in folder_path.rglob("*.png"))
                    logger.info(f"✓ Found {img_count} images in {folder_name}")
                    return True
            
            logger.warning("Data folder not found in expected locations")
            return False
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to clone repository: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error downloading dataset: {e}")
            return False
    
    def download_thanhsan_dataset(self) -> bool:
        """
        Download helmet violation detection dataset.
        
        Dataset contains:
        - Helmet vs no-helmet annotations
        - Rider pose variations
        - Different helmet types
        - Multi-angle views
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 3: ThanhSan97 - Helmet Violation Detection")
        logger.info("="*70)
        
        try:
            repo_url = "https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16"
            dest_path = self.output_dir / "thanhsan_helmet_detection"
            
            if dest_path.exists():
                logger.info(f"Dataset already exists at {dest_path}")
                return True
            
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dest_path)],
                check=True,
                capture_output=True
            )
            
            # Look for dataset
            data_folders = ["data", "dataset", "datasets"]
            for folder_name in data_folders:
                folder_path = dest_path / folder_name
                if folder_path.exists():
                    img_count = sum(1 for f in folder_path.rglob("*.jpg"))
                    img_count += sum(1 for f in folder_path.rglob("*.png"))
                    logger.info(f"✓ Found {img_count} images in {folder_name}")
                    return True
            
            logger.warning("Data folder not found in expected locations")
            return False
        
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to clone repository: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Error downloading dataset: {e}")
            return False
    
    def download_ccpd_dataset(self) -> bool:
        """
        Download CCPD (Chinese City Parking Dataset) - 290K license plate images.
        
        Large dataset (3.5GB+) - Public and challenging dataset for plate detection.
        Manual download recommended due to size.
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 4: CCPD - Chinese City Parking (290K images, 3.5GB+)")
        logger.info("="*70)
        
        logger.info("CCPD is too large to auto-download (3.5GB+)")
        logger.info("Manual download recommended:")
        logger.info("  1. Visit: https://github.com/detectRecog/CCPD")
        logger.info("  2. Download: CCPD-Base (includes: normal, weather, illumination, blur)")
        logger.info("  3. Extract to: ./datasets/CCPD")
        logger.info("  4. Run: python prepare_datasets.py --convert-ccpd")
        
        return False
    
    def download_ridesafe_references(self) -> bool:
        """
        Download RideSafe-400 references and documentation.
        
        RideSafe-400 contains 354K motorcycle-rider annotations.
        Full dataset requires direct request, but documentation is available.
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 5: RideSafe-400 References (354K annotations)")
        logger.info("="*70)
        
        logger.info("RideSafe-400 is restricted dataset requiring direct request")
        logger.info("To obtain:")
        logger.info("  1. Visit: https://ridesafe.org/")
        logger.info("  2. Request access with research/commercial purpose")
        logger.info("  3. Download full dataset")
        logger.info("  4. Extract to: ./datasets/RideSafe")
        
        return False
    
    def download_public_samples(self) -> bool:
        """
        Download small public samples for quick testing.
        """
        logger.info("\n" + "="*70)
        logger.info("DATASET 6: Public Samples (Quick Testing)")
        logger.info("="*70)
        
        samples_dir = self.output_dir / "public_samples"
        samples_dir.mkdir(exist_ok=True)
        
        logger.info(f"Sample data directory created: {samples_dir}")
        logger.info("Add your test images here for quick validation")
        
        return True
    
    def summarize_datasets(self) -> dict:
        """Generate summary of downloaded datasets."""
        summary = {}
        
        for dataset_dir in self.output_dir.iterdir():
            if dataset_dir.is_dir():
                img_count = sum(1 for f in dataset_dir.rglob("*.jpg"))
                img_count += sum(1 for f in dataset_dir.rglob("*.png"))
                txt_count = sum(1 for f in dataset_dir.rglob("*.txt"))
                
                summary[dataset_dir.name] = {
                    'images': img_count,
                    'labels': txt_count,
                    'path': str(dataset_dir)
                }
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download datasets from GitHub repositories for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Download all available datasets
  python download_github_datasets.py --all
  
  # Download specific dataset
  python download_github_datasets.py --kashishparmar02
  python download_github_datasets.py --ronlek
  python download_github_datasets.py --thanhsan
  
  # Show summary only
  python download_github_datasets.py --summary
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Download all available datasets')
    parser.add_argument('--kashishparmar02', action='store_true',
                       help='Download KashishParmar02 - Triple Rider Dataset (6000+ images)')
    parser.add_argument('--ronlek', action='store_true',
                       help='Download RonLek - ALPR Indian Vehicles (2500+ images)')
    parser.add_argument('--thanhsan', action='store_true',
                       help='Download ThanhSan97 - Helmet Detection Dataset')
    parser.add_argument('--ccpd', action='store_true',
                       help='Instructions for CCPD dataset (manual, 3.5GB)')
    parser.add_argument('--ridesafe', action='store_true',
                       help='Instructions for RideSafe-400 dataset (manual)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary of available datasets')
    parser.add_argument('--output', type=str, default='./datasets',
                       help='Output directory for datasets (default: ./datasets)')
    
    args = parser.parse_args()
    
    downloader = GitHubDatasetDownloader(output_dir=args.output)
    
    logger.info("="*70)
    logger.info("GitHub Dataset Downloader for Traffic Violation Detection")
    logger.info("="*70 + "\n")
    
    # Download requested datasets
    results = {}
    
    if args.all or args.kashishparmar02:
        results['kashishparmar02'] = downloader.download_kashishparmar02_dataset()
    
    if args.all or args.ronlek:
        results['ronlek'] = downloader.download_ronlek_dataset()
    
    if args.all or args.thanhsan:
        results['thanhsan'] = downloader.download_thanhsan_dataset()
    
    if args.all or args.ccpd:
        results['ccpd'] = downloader.download_ccpd_dataset()
    
    if args.all or args.ridesafe:
        results['ridesafe'] = downloader.download_ridesafe_references()
    
    if args.all:
        results['public_samples'] = downloader.download_public_samples()
    
    # Show summary
    if args.all or args.summary:
        logger.info("\n" + "="*70)
        logger.info("DATASET SUMMARY")
        logger.info("="*70)
        
        summary = downloader.summarize_datasets()
        
        total_images = 0
        total_labels = 0
        
        for dataset_name, info in summary.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Path: {info['path']}")
            logger.info(f"  Images: {info['images']}")
            logger.info(f"  Labels: {info['labels']}")
            total_images += info['images']
            total_labels += info['labels']
        
        logger.info(f"\nTotal Images: {total_images}")
        logger.info(f"Total Labels: {total_labels}")
        
        if total_images > 0:
            logger.info(f"\n✓ Ready for training with {total_images} images!")
            logger.info("\nNext steps:")
            logger.info("  1. Run: python prepare_datasets.py")
            logger.info("  2. Then: python train_quick.py --pipeline full")


if __name__ == "__main__":
    main()
