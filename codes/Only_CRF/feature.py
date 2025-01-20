import cv2
import numpy as np
from skimage.feature import canny
from scipy.ndimage import gaussian_filter
import os

root_dir = "../../dataset/test/image"
for image_path_ in os.listdir(root_dir):
    # 1. 加载图像
    image_path = os.path.join(root_dir, image_path_)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. 提取颜色对比度特征
    def color_contrast(image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_channel = lab_image[:, :, 0] 
        contrast = np.abs(l_channel - np.mean(l_channel))
        return contrast / contrast.max()

    contrast_feature = color_contrast(image)

    # 3. 提取边缘特征
    def edge_feature(image, sigma=2):
        edges = canny(image, sigma=sigma)
        return edges.astype(np.float32)

    edge_feature_map = edge_feature(gray_image)

    # 4. 线性融合特征
    alpha = 0.6  # 调节颜色和边缘特征的影响
    probabilities = alpha * contrast_feature + (1 - alpha) * edge_feature_map
    probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())  # 归一化

    # 5. 高斯滤波优化
    probabilities_smoothed = gaussian_filter(probabilities, sigma=3)

    # 6. 保存结果到文件
    im_name = image_path_.split('.')[0]
    output_dir = os.path.join("./Only_CRF", im_name)
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "original_image.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 保存原图
    cv2.imwrite(os.path.join(output_dir, "color_contrast_feature.png"), (contrast_feature * 255).astype(np.uint8))  # 保存颜色特征图
    cv2.imwrite(os.path.join(output_dir, "edge_feature.png"), (edge_feature_map * 255).astype(np.uint8))  # 保存边缘特征图
    cv2.imwrite(os.path.join(output_dir, "probability_map.png"), (probabilities_smoothed * 255).astype(np.uint8))  # 保存概率图

    print(f"All images have been saved to {output_dir}")
