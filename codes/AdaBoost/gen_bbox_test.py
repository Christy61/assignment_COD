import cv2
import numpy as np

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

mask = cv2.imread("../../dataset/train/GT/COD10K-CAM-2-Terrestrial-47-Tiger-2908.png", cv2.IMREAD_GRAYSCALE)
bbox = mask_to_bbox(mask)

output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
print("Bounding Box:", bbox)
cv2.imwrite("test1.png", output_image)