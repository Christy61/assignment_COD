# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd
import cv2
join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
from PIL import Image

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default="data/custom/train/image",
    help="path to the images",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default="data/custom/train/GT",
    help="path to the ground truth (gt)",
)


parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--npz_path",
    type=str,
    default="data/Train",
    help="path to save the npz files",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="custom",
    help="dataset name; used to name the final npz file, e.g., demo2d.npz",
)
parser.add_argument("--image_size", type=int, default=256, help="image size")
parser.add_argument(
    "--img_name_suffix", type=str, default=".jpg", help="image name suffix"
)
parser.add_argument("--label_id", type=int, default=255, help="label id")
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="work_dir_cod/SAM/sam_vit_b_01ec64.pth",
    help="checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:2", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# convert 2d grey or rgb images to npz file
imgs = []
gts = []
number = []
boundary = []
img_embeddings = []
global num_of_processed_imgs
num_of_processed_imgs = 0


sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
    args.device
)

save_path = args.npz_path + "_" + args.model_type
os.makedirs(save_path, exist_ok=True)
def find_bundary(img, mask, path, name):
    mask = mask * 255 # convert to 0-255
    kernel = np.ones((3, 3), dtype=np.uint8)
    fore = cv2.dilate(mask, kernel, 3)
    dilate = (fore - mask)
    kernel_again = np.ones((5, 5), dtype=np.uint8)
    dilate_again = cv2.dilate(dilate, kernel_again, 3)

    edges = cv2.Canny(img, 0.2, 0.6)
    os.makedirs(os.path.join(path + '/dilate/'), exist_ok=True)
    os.makedirs(os.path.join(path + '/boundary/'), exist_ok=True)
    os.makedirs(os.path.join(path + '/canny/'), exist_ok=True)
    cv2.imwrite(os.path.join(path + '/dilate/', name), fore)
    cv2.imwrite(os.path.join(path + '/boundary/', name), dilate_again)
    cv2.imwrite(os.path.join(path + '/canny/', name), edges)

    edges_2 = Image.open(os.path.join(os.path.join(path + '/canny/', name))).convert('1')
    dilate_again_2 = Image.open(os.path.join(os.path.join(path + '/boundary/', name))).convert('1')

    boundary_grads = (Image.fromarray(np.array(edges_2) * np.array(dilate_again_2)))
    os.makedirs(os.path.join(path + '/boundary_grads/'), exist_ok=True)
    boundary_grads.save(os.path.join(os.path.join(path + '/boundary_grads/', gt_name)))

    return boundary_grads

def process(gt_name: str, image_name: str, num_of_processed_imgs:int):
    if image_name == None:
        image_name = gt_name.split(".")[0] + args.img_name_suffix  # Find the name of images based on the name of GT
    gt_data = io.imread(join(args.gt_path, gt_name))
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truch image
    gt_data = transform.resize(
        gt_data == args.label_id,
        (args.image_size, args.image_size),
        order=0,
        preserve_range=True,
        mode="constant",
    )
    # convert to uint8
    gt_data = np.uint8(gt_data)

    if np.sum(gt_data) > 5:  # exclude tiny objects(Polyps may be small in shape)
        """Optional binary thresholding can be added"""
        assert (
            np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2
        ), "ground truth should be binary"

        image_data = io.imread(join(args.img_path, image_name))
        # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(
            image_data, 99.5
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # min-max normalize and scale
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = transform.resize(
            image_data_pre,
            (args.image_size, args.image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        image_data_pre = np.uint8(image_data_pre)

        imgs.append(image_data_pre)
        number.append(image_name)

        print("the number of images: " + str(len(imgs)) + " and the name of image: " + str(image_name))
        num_of_processed_imgs = num_of_processed_imgs + 1
        assert np.sum(gt_data) > 5, "ground truth should have more than 50 pixels"

        gts.append(gt_data)
        boundary.append(find_bundary(image_data_pre, gt_data, save_path+'/'+args.data_name, gt_name))

        # resize image to 3*1024*1024
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
            args.device
        )
        input_image = sam_model.preprocess(
            resize_img_tensor[None, :, :, :]
        )  # (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        # pre-compute the image embedding
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])

    return num_of_processed_imgs


if args.csv != None:

    try:
        os.path.exists(args.csv)
    except FileNotFoundError as e:
        print(f"File {args.csv} not found!!")

    df = pd.read_csv(args.csv)
    bar = tqdm(df.iterrows(), total=len(df))
    for idx, row in bar:
        process(row.mask_filename, row.image_filename)

else:

    names = sorted(os.listdir(args.gt_path))
    # print the number of images found in the ground truth folder
    print("image number:", len(names))
    for gt_name in tqdm(names):
        num_of_processed_imgs = process(gt_name, None, num_of_processed_imgs)
    print("the number of processed images: " + str(num_of_processed_imgs))


# stack the list to array
print("Num. of images:", len(imgs))
if len(imgs) > 1:
    imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
    gts = np.stack(gts, axis=0)  # (n, 256, 256)
    img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
    boundary = np.stack(boundary, axis=0)  # (n, 256, 256)
    np.savez_compressed(
        join(save_path, args.data_name + ".npz"),
        imgs=imgs,
        gts=gts,
        number=number,
        img_embeddings=img_embeddings,
        boundary=boundary
    )
    # save an example image for sanity check
    idx = np.random.randint(imgs.shape[0])
    img_idx = imgs[idx, :, :, :]
    gt_idx = gts[idx, :, :]
    bd = segmentation.find_boundaries(gt_idx, mode="inner")
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(save_path + ".png", img_idx, check_contrast=False)
else:
    print(
        "Do not find image and ground-truth pairs. Please check your dataset and argument settings"
    )