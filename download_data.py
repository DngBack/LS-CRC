import os
import numpy as np
from datasets import load_dataset
from PIL import Image


def _detach_pil(img):
    """Load pixels and copy so save() is not tied to HF/PIL PNG decoder internals."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    img.load()
    return img.copy()


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

        img = _detach_pil(img)
        mask = _detach_pil(mask)
        img.save(img_path)
        mask.save(mask_path)
    print(f"Saved {len(ds)} images to {output_dir}")
    return len(ds)


def generate_splits(output_dir, n_train, n_val, n_cal, n_total, min_test_frac=0.10):
    import random

    if n_total < 4:
        raise ValueError(f"n_total must be at least 4, got {n_total}")

    # Reserve at least ~10% (min 1) for test so evaluation is not trivial.
    min_test = max(1, int(round(n_total * min_test_frac)))
    max_tvc = n_total - min_test

    # Shrink train/val/cal until they fit in max_tvc (train shrinks first).
    while n_train + n_val + n_cal > max_tvc:
        if n_train > max(n_val, n_cal, 1):
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        elif n_cal > 1:
            n_cal -= 1
        else:
            raise ValueError(
                f"Cannot fit train/val/cal into n_total={n_total} "
                f"while keeping at least {min_test} test samples."
            )

    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    cal_idx = indices[n_train + n_val : n_train + n_val + n_cal]
    test_idx = indices[n_train + n_val + n_cal :]
    
    def write_split(split, ids):
        with open(os.path.join(output_dir, f"{split}.txt"), 'w') as f:
            for i in ids:
                f.write(f"{i:04d}.png\n")
                
    write_split('train', train_idx)
    write_split('val', val_idx)
    write_split('cal', cal_idx)
    write_split('test', test_idx)
    print(
        f"Generated splits for {output_dir}: train={n_train} val={n_val} "
        f"cal={n_cal} test={len(test_idx)} (n_total={n_total})"
    )


def regenerate_splits_from_disk(output_dir, n_train=400, n_val=50, n_cal=50):
    """
    Rebuild train/val/cal/test txt files from whatever images exist in output_dir/images.
    Use this if splits were generated with a wrong n_total (missing files on disk).
    """
    img_dir = os.path.join(output_dir, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"No images directory: {img_dir}")
    n_total = len(
        [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if n_total < 4:
        raise ValueError(f"Need at least 4 images in {img_dir}, found {n_total}")
    generate_splits(output_dir, n_train, n_val, n_cal, n_total)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HF medical segmentation data or regenerate splits.")
    parser.add_argument(
        "--regenerate-splits",
        nargs="+",
        metavar="DIR",
        help="Only rewrite split txt files for these dataset roots (e.g. data/CVC-ClinicDB), using files on disk.",
    )
    parser.add_argument("--train", type=int, default=None, help="Train count for --regenerate-splits (default: 400 or 500 for Kvasir-sized dirs).")
    parser.add_argument("--val", type=int, default=50, help="Val count for --regenerate-splits.")
    parser.add_argument("--cal", type=int, default=50, help="Cal count for --regenerate-splits.")
    args = parser.parse_args()

    if args.regenerate_splits:
        for root in args.regenerate_splits:
            nt = args.train
            if nt is None:
                nt = 500 if "Kvasir" in root else 400
            regenerate_splits_from_disk(root, n_train=nt, n_val=args.val, n_cal=args.cal)
    else:
        try:
            n_kv = download_and_save("kowndinya23/Kvasir-SEG", "train", "data/Kvasir-SEG")
            generate_splits("data/Kvasir-SEG", 500, 80, 150, n_kv)
        except Exception as e:
            print("Failed Kvasir-SEG:", e)

        try:
            n_cvc = download_and_save("Angelou0516/CVC-ClinicDB", "train", "data/CVC-ClinicDB")
            generate_splits("data/CVC-ClinicDB", 400, 50, 50, n_cvc)
        except Exception as e:
            print("Failed CVC:", e)
