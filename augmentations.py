import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import random

INPUT_DIR = "./data/UE4CD/"
OUTPUT_DIR = "./data/UE4CD_augmented/"
MIN_OBJ_AREA = 100

os.makedirs(f"{OUTPUT_DIR}/time1", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/time2", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/mask", exist_ok=True)

photo_aug = A.OneOf([
    A.RandomBrightnessContrast(),
    A.GaussNoise(),
    A.MotionBlur(blur_limit=5),
], p=1.0)

def remove_small_objects(mask, min_area=100):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return cleaned

def occlude_random_region(img, mask, max_size=50):
    h, w = img.shape[:2]
    for _ in range(10):
        x = np.random.randint(0, w - max_size)
        y = np.random.randint(0, h - max_size)
        roi = mask[y:y+max_size, x:x+max_size]
        if np.sum(roi) == 0:
            img[y:y+max_size, x:x+max_size] = np.random.randint(0, 255, (max_size, max_size, 3), dtype=np.uint8)
            break
    return img

file_names = sorted(os.listdir(f"{INPUT_DIR}/time1"))

for fname in tqdm(file_names):
    name, ext = os.path.splitext(fname)

    img1 = cv2.imread(os.path.join(INPUT_DIR, "time1", fname))
    img2 = cv2.imread(os.path.join(INPUT_DIR, "time2", fname))
    mask = cv2.imread(os.path.join(INPUT_DIR, "mask", fname), cv2.IMREAD_GRAYSCALE)

    # === 1. Photometric only ===
    img1_photo = photo_aug(image=img1)['image']
    img2_photo = photo_aug(image=img2)['image']

    cv2.imwrite(os.path.join(OUTPUT_DIR, "time1", f"{name}_photometric{ext}"), img1_photo)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "time2", f"{name}_photometric{ext}"), img2_photo)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask",  f"{name}_photometric{ext}"), mask)

    # === 2. Cleaned mask only ===
    cleaned_mask = remove_small_objects(mask, min_area=MIN_OBJ_AREA)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "time1", f"{name}_cleaned{ext}"), img1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "time2", f"{name}_cleaned{ext}"), img2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask",  f"{name}_cleaned{ext}"), cleaned_mask * 255)

    # === 3. Occlusion only ===
    img2_occluded = occlude_random_region(img2.copy(), mask)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "time1", f"{name}_occluded{ext}"), img1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "time2", f"{name}_occluded{ext}"), img2_occluded)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask",  f"{name}_occluded{ext}"), mask)
