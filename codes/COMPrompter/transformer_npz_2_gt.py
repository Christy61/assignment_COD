import numpy as np
from skimage import io
import os
join = os.path.join
import cv2 as cv

path = "./output/predict_image/custom_vit_b"
save_path = './output/predict_image/'
probs_save_path = './output/probs/'
npz_folders = sorted(os.listdir(path))
for npz_folder in npz_folders:
    data = np.load(join(path, npz_folder,"custom") + '.npz')
    imgs = data["medsam_segs"]  # 假设图像数据保存在名为"imgs"的数组中
    name = data['number']
    probs = data["medsam_seg_prob"]

    # 将图像数据转换为正确的数据类型和范围
    imgs = imgs.astype(np.uint8)
    imgs = (imgs * 255.0).astype(np.uint8)  # 根据图像数据的范围进行调整
    medsam_seg_prob = (probs * 255).astype(np.uint8)

    for i, img in enumerate(imgs):
        num = 149 + i
        img_path = join(save_path, npz_folder) + "/" + name[i]  # 设置保存图片的路径和文件名
        img_path = img_path.replace("jpg", "png")
        if not os.path.exists(join(save_path, npz_folder)):
            os.makedirs(join(save_path, npz_folder), exist_ok=True)
        io.imsave(img_path, img)

    print("Images saved successfully.")

    # for i, img in enumerate(medsam_seg_prob):
    #     num = 149 + i
    #     img_path = join(probs_save_path, npz_folder) + "/" + name[i]  # 设置保存图片的路径和文件名
    #     img_path = img_path.replace("jpg", "png")
    #     if not os.path.exists(join(probs_save_path, npz_folder)):
    #         os.makedirs(join(probs_save_path, npz_folder), exist_ok=True)
    #     # io.imsave(img_path, img)
    #     cv.imwrite(img_path, img)

    # print("Probs saved successfully.")