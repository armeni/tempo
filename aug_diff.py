import cv2
import os
import matplotlib.pyplot as plt

def show_augmented_comparison(filename="1.png"):
    original_path = "data"
    augmented_path = "augmented_data"

    img1_orig = cv2.imread(os.path.join(original_path, "time1", filename))
    img2_orig = cv2.imread(os.path.join(original_path, "time2", filename))
    mask_orig = cv2.imread(os.path.join(original_path, "mask", filename), cv2.IMREAD_GRAYSCALE)

    img1_aug = cv2.imread(os.path.join(augmented_path, "time1", filename))
    img2_aug = cv2.imread(os.path.join(augmented_path, "time2", filename))
    mask_aug = cv2.imread(os.path.join(augmented_path, "mask", filename), cv2.IMREAD_GRAYSCALE)

    img1_orig = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
    img2_orig = cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB)
    img1_aug = cv2.cvtColor(img1_aug, cv2.COLOR_BGR2RGB)
    img2_aug = cv2.cvtColor(img2_aug, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].imshow(img1_orig)
    axs[0, 0].set_title("Original Time1")
    axs[0, 1].imshow(img2_orig)
    axs[0, 1].set_title("Original Time2")
    axs[0, 2].imshow(mask_orig, cmap='gray')
    axs[0, 2].set_title("Original Mask")

    axs[1, 0].imshow(img1_aug)
    axs[1, 0].set_title("Augmented Time1")
    axs[1, 1].imshow(img2_aug)
    axs[1, 1].set_title("Augmented Time2")
    axs[1, 2].imshow(mask_aug, cmap='gray')
    axs[1, 2].set_title("Augmented Mask")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

show_augmented_comparison("1.png")  