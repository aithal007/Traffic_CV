
import os
import shutil

def merge_datasets(target_name, source_mappings, output_dir):
    print(f"--- Creating {target_name} Dataset ---")
    img_out = os.path.join(output_dir, 'images')
    lbl_out = os.path.join(output_dir, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    file_count = 0
    for folder_rel_path, class_map in source_mappings.items():
        base = os.path.join('datasets', 'raw', folder_rel_path)
        if not os.path.exists(base):
            print(f"Skipping missing: {base}")
            continue
            
        # Search for images/labels subfolders (recursive)
        for root, dirs, files in os.walk(base):
            if 'images' in root.lower():
                label_root = root.replace('images', 'labels').replace('Images', 'Labels')
                if not os.path.exists(label_root): continue
                
                print(f"Processing: {root}")
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, f)
                        lbl_name = os.path.splitext(f)[0] + '.txt'
                        lbl_path = os.path.join(label_root, lbl_name)
                        
                        if os.path.exists(lbl_path):
                            # Copy image with unique prefix to avoid collisions
                            safe_folder = folder_rel_path.replace('/', '_').replace(' ', '_')
                            new_name = f"{safe_folder}_{f}"
                            shutil.copy(img_path, os.path.join(img_out, new_name))
                            
                            # Process and copy label
                            with open(lbl_path, 'r') as lf:
                                lines = lf.readlines()
                            
                            new_lines = []
                            for line in lines:
                                parts = line.strip().split()
                                if not parts: continue
                                try:
                                    old_id = int(parts[0])
                                    if old_id in class_map:
                                        new_id = class_map[old_id]
                                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                                except:
                                    continue
                            
                            with open(os.path.join(lbl_out, os.path.splitext(new_name)[0] + '.txt'), 'w') as lf:
                                lf.writelines(new_lines)
                            file_count += 1
    print(f"Finished {target_name}. Total images: {file_count}")

# 1. HELMET MAPPING (0=with, 1=no)
helmet_sources = {
    'helmet_dataset': {0:0, 1:1},
    'helmet_dataset_large': {2:0, 3:1},
    'helmet_detection_1': {0:0, 1:0, 2:1, 3:0, 4:0}, 
    'helmet_detection_2': {0:0, 1:1},
    'helmet_final_refined': {0:0, 3:1},
    'fog_weather_data': {3:0, 5:1}
}

# 2. BIKE MAPPING (0=motorcycle)
bike_sources = {
    'bike_dataset': {0:0},
    'vehical_detection_1': {0:0},
    'triple_dataset': {0:0},
    'triple_sharing': {1:0}, 
    'helmet_final_refined': {2:0},
    'fog_weather_data': {4:0}
}

base_dir = r'c:\Users\Lenovo\Documents\cv_project'
os.chdir(base_dir)

merge_datasets("MASTER_HELMET", helmet_sources, 'datasets/MASTER_HELMET')
merge_datasets("MASTER_BIKE", bike_sources, 'datasets/MASTER_BIKE')
