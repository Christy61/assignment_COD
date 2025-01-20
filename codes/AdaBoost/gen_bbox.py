import cv2
import numpy as np
import os
from train import visualize_bbox

def mask_to_bbox(mask):
    """
    Converts a split mask to a bounding box
    """
    # Find the outer rectangle of the target area
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Returns the coordinates and width of the bounding box
    return [x_min, y_min, x_max - x_min, y_max - y_min]

test_path = f"../../dataset/test/GT"
os.makedirs("gt/bbox", exist_ok=True)

for img_path in os.listdir(test_path):
    img_name = img_path.split(".")[0]
    output_txt_path = f"gt/bbox/{img_name}.txt"
    mask = cv2.imread(os.path.join(test_path, img_path), cv2.IMREAD_GRAYSCALE)
    bbox = mask_to_bbox(mask)
    image_rgb_path = os.path.join(test_path, img_path)
    image_rgb_path = image_rgb_path.replace("png", "jpg")
    image_rgb_path = image_rgb_path.replace("GT", "image")
    image = cv2.imread(image_rgb_path)

    with open(output_txt_path, "w") as txt_file:
        txt_file.write(f"object {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    file_path = f"gt/{img_name}.png"
    visualize_bbox(image, file_path, bbox, title=f"Test Image {img_name} - GT Bounding Box")