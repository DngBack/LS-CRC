import os
from datasets import load_dataset
from PIL import Image

def download_and_save(dataset_name, split, output_dir):
    print(f"Downloading {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split)
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    for i, item in enumerate(ds):
        # 'image' and 'mask' or 'label' depending on HF dataset
        if 'image' in item and 'label' in item:
            img = item['image']
            mask = item['label']
        elif 'image' in item and 'mask' in item:
            img = item['image']
            mask = item['mask']
        elif 'image' in item and 'annotation' in item:
            img = item['image']
            mask = item['annotation']
        else:
            print("Unknown keys:", item.keys())
            continue
            
        img_path = os.path.join(output_dir, "images", f"{i:04d}.png")
        mask_path = os.path.join(output_dir, "masks", f"{i:04d}.png")
        
        img.save(img_path)
        mask.save(mask_path)
    print(f"Saved {len(ds)} images to {output_dir}")

def generate_splits(output_dir, n_train, n_val, n_cal, n_total):
    indices = list(range(n_total))
    import random
    random.seed(42)
    random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    cal_idx = indices[n_train+n_val:n_train+n_val+n_cal]
    test_idx = indices[n_train+n_val+n_cal:]
    
    def write_split(split, ids):
        with open(os.path.join(output_dir, f"{split}.txt"), 'w') as f:
            for i in ids:
                f.write(f"{i:04d}.png\n")
                
    write_split('train', train_idx)
    write_split('val', val_idx)
    write_split('cal', cal_idx)
    write_split('test', test_idx)
    print(f"Generated splits for {output_dir}")

if __name__ == "__main__":
    try:
        download_and_save("kowndinya23/Kvasir-SEG", "train", "data/Kvasir-SEG")
        generate_splits("data/Kvasir-SEG", 500, 80, 150, 880)
    except Exception as e:
        print("Failed Kvasir-SEG:", e)
        
    try:
        download_and_save("Angelou0516/CVC-ClinicDB", "train", "data/CVC-ClinicDB")
        generate_splits("data/CVC-ClinicDB", 400, 50, 50, 612)
    except Exception as e:
        print("Failed CVC:", e)
