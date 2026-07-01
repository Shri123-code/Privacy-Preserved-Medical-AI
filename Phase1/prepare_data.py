import os
import shutil
import random


SOURCE_PATH = "chest_xray/train" 
TARGET_BASE_DIR = "federated_data"
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]
CATEGORIES = ["NORMAL", "PNEUMONIA"]

def setup_hospitals():
    
    for hosp in HOSPITALS:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(TARGET_BASE_DIR, hosp, cat), exist_ok=True)
    
    print("Step 1: Folders created.")

    
    for cat in CATEGORIES:
        source_cat_dir = os.path.join(SOURCE_PATH, cat)
        images = os.listdir(source_cat_dir)
        random.shuffle(images) 
        
        
        avg = len(images) // 4
        
        for i, hosp in enumerate(HOSPITALS):
            
            start = i * avg
            end = (i + 1) * avg if i != 3 else len(images)
            chunk = images[start:end]
            
            print(f"Step 2: Copying {len(chunk)} {cat} images to {hosp}...")
            
            for img_name in chunk:
                src_file = os.path.join(source_cat_dir, img_name)
                dest_file = os.path.join(TARGET_BASE_DIR, hosp, cat, img_name)
                
        
                shutil.copy(src_file, dest_file) 

if __name__ == "__main__":
    if not os.path.exists(SOURCE_PATH):
        print(f"ERROR: Could not find your dataset at {SOURCE_PATH}")
    else:
        setup_hospitals()
        print("✅ SUCCESS: All 4 hospital silos are ready for Federated Learning!")