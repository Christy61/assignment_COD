import os
import cv2 as cv
import imageio

if __name__ == '__main__':
    pros_path_dir = "./output/probs/custom_vit_b"
    img_dir = "../../dataset/test/image"
    for name in os.listdir(img_dir):
        image_path = img_dir + "/" + name
        label_path = image_path.replace("image", "GT")
        label_path = label_path.replace("jpg", "png")
        pros_path = pros_path_dir + "/" + name
        pros_path = pros_path.replace("jpg", "png")
        print(pros_path)
        # Load data
        label = imageio.imread(label_path)
        pros_ = imageio.imread(pros_path)
        H, W = label.shape
        pros_resized = cv.resize(pros_, (W, H), interpolation=cv.INTER_CUBIC)
        imageio.imsave(pros_path, pros_resized)

        
