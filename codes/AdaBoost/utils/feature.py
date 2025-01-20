import numpy as np
from skimage.feature import hog
import cv2 as cv
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings(action='ignore')
from tqdm import tqdm
import torch
from sklearn.decomposition import PCA

def prepare_data_for_adaboost(loader, model, device, window_size=(32, 32), stride=16, threshold=0.3):
    
    print("prepare dataset...")
    features = []
    labels = []
    
    for images, lbls in tqdm(loader):
        images = images.to(device) 
        lbls = lbls.to(device) 
        for img, lbl in zip(images, lbls):
            img_np = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
            lbl_np = lbl.cpu().numpy().squeeze()  # Convert to HxW
            h, w = img_np.shape[:2]
            
            all_window_features = []
            all_window_labels = []

            # Sliding window for each scale
            for y in range(0, h - window_size[1] + 1, stride):
                for x in range(0, w - window_size[0] + 1, stride):
                    window = img_np[y:y + window_size[1], x:x + window_size[0]]
                    mask_window = lbl_np[y:y + window_size[1], x:x + window_size[0]]
                    all_window_features.append(window)
                    
                    foreground_ratio = np.sum(mask_window) / mask_window.size
                    label = 1 if foreground_ratio >= threshold else 0
                    all_window_labels.append(label)

        if len(all_window_features) > 0:
                batch_features = np.array(all_window_features)
                batch_features = torch.tensor(batch_features).permute(0, 3, 1, 2)  # Convert to (N, C, H, W) format
                batch_features = batch_features.to(device)  # Move to device

                feature_batch = extract_features_with_resnet(model, batch_features, device)
                features.extend(feature_batch)
                labels.extend(all_window_labels)

    return np.array(features), np.array(labels)



def reduce_dimensionality_with_pca(features, n_components=512):
    """
    Dimensionality reduction using PCA
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def extract_hog_features(image, resize=(64, 64)):
    """
    generate HOG feature
    """

    if image.ndim == 3:  # Check if the image is multichannel
        image = rgb2gray(image)
    image_resized = cv.resize(image, resize)
    features, _ = hog(image_resized, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    return features


def extract_features_with_resnet(model, images, device):
    """
    generate feature from resnet50
    """
    model.eval()  
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    
    return features.cpu().numpy()
