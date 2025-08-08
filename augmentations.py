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

import numpy as np
import cv2
import random

def occlude_random_region(img, mask, min_size=50, max_size=250, max_attempts=20):
    h, w = img.shape[:2]
    occluded_img = img.copy()

    mean_color = np.mean(img[mask == 0], axis=0)  
    mean_color = mean_color.astype(np.uint8)

    for _ in range(max_attempts):
        shape_type = random.choice(["rect", "circle"])
        size_x = random.randint(min_size, max_size)
        size_y = random.randint(min_size, max_size) if shape_type == "rect" else size_x

        x = random.randint(0, w - size_x)
        y = random.randint(0, h - size_y)

        roi_mask = mask[y:y+size_y, x:x+size_x]
        if np.sum(roi_mask) != 0:
            continue

        color_variation = np.random.randint(-20, 20, 3)
        occlusion_color = np.clip(mean_color + color_variation, 0, 255).astype(np.uint8).tolist()

        if shape_type == "rect":
            cv2.rectangle(occluded_img, (x, y), (x + size_x, y + size_y), occlusion_color, -1)
        elif shape_type == "circle":
            center = (x + size_x // 2, y + size_y // 2)
            radius = min(size_x, size_y) // 2
            cv2.circle(occluded_img, center, radius, occlusion_color, -1)

        break 

    return occluded_img

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
