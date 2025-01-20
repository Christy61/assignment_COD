import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import augmentation
from utils.dataloader import Custom
from utils.feature import prepare_data_for_adaboost, extract_features_with_resnet
import torch
from utils.argument import get_parser
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import cv2 as cv

def visualize_bbox(image, file_path, bbox, title="Detected Bounding Box"):

    cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    cv.imwrite(file_path, image)

def load_dataset(base_dir):
    # data enhancement
    train_tf = augmentation.Mask_Aug(transforms=[augmentation.ToTensor(), augmentation.PILToTensor(),
                                                    augmentation.Resize((512, 512)),
                                                    augmentation.RandomCrop((384, 384)),
                                                    augmentation.RandomHorizontalFlip(),
                                                    augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_tf = augmentation.Mask_Aug(transforms=[augmentation.ToTensor(), augmentation.PILToTensor(),
                                                augmentation.Resize((384, 384)),
                                                augmentation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    # create Dataset
    train_dataset = Custom(base_dir=base_dir, split="train", transform=train_tf)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Custom(base_dir=base_dir, split="test", transform=val_tf)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    files_name = val_dataset.get_name()
    return train_loader, val_loader, files_name


class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = X
        self.y = y
        self.weights = weights

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]


class SmallFCN(nn.Module):
    def __init__(self, input_dim):
        super(SmallFCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, weights):
        self.X = X
        self.y = y
        self.weights = weights

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx]

    
class AdaBoostClassifier:
    def __init__(self, n_estimators=10, input_dim=1000, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        self.input_dim = input_dim

    def fit(self, X_train, y_train, save_path="best_adaboost_model.pth"):
        n_samples = X_train.shape[0]
        weights = np.ones(n_samples) / n_samples  
        best_accuracy = 0.0

        for i in range(self.n_estimators):
            # Creating weighted data sets
            sampler = WeightedRandomSampler(weights, len(weights))
            dataset = SlidingWindowDataset(X_train, y_train, weights)
            data_loader = DataLoader(dataset, batch_size=64, sampler=sampler)

            # Define the base learner
            model = SmallFCN(self.input_dim).cuda()
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            model.train()
            for epoch in tqdm(range(75)): 
                for X_batch, y_batch, _ in data_loader:
                    X_batch, y_batch = X_batch.float().cuda(), y_batch.float().cuda()
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # Access to forecast results
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.tensor(X_train, dtype=torch.float32).cuda()).cpu().numpy().squeeze()
                y_pred = (y_pred >= 0.5).astype(int)

            # Calculate accuracy
            accuracy = np.mean(y_pred == y_train)
            print(f"Estimator {i + 1}/{self.n_estimators}, Accuracy: {accuracy:.4f}")

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'alpha': self.alphas,
                    'accuracy': best_accuracy
                }, save_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")

            # Calculation of the margin of error
            incorrect = (y_pred != y_train).astype(int)
            error = np.dot(weights, incorrect) / np.sum(weights)

            # Calculate the weights of the model alpha
            if error > 0.5:  # If the error rate is greater than 0.5, stop training
                break
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update sample weights
            weights = weights * np.exp(-alpha * (1 - 2 * incorrect))
            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Combining the weighted predictions of all base learners
        final_pred = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            with torch.no_grad():
                pred = model(torch.tensor(X, dtype=torch.float32).cuda()).cpu().numpy().squeeze()
                pred = (pred >= 0.5).astype(int)
            final_pred += alpha * pred
        return (final_pred >= 0.4).astype(int)

def predict_bounding_boxes(test_loader, model, adaboost, device, window_size=(32, 32), stride=16):
    model.eval()
    all_bboxes = []

    for images, lbls in tqdm(test_loader, desc="Predicting bounding boxes"):
        images = images.to(device) 
        for image in images:
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            h, w = img_np.shape[:2]
            
            detected_bboxes = []
            all_window_features = []
            features = []
            windows_loc = []
            
            for y in range(0, h - window_size[1] + 1, stride):
                for x in range(0, w - window_size[0] + 1, stride):
                    window = img_np[y:y + window_size[1], x:x + window_size[0]] 
                    all_window_features.append(window)
                    windows_loc.append([x, y, x + window_size[0], y + window_size[1]])

            if len(all_window_features) > 0:
                batch_features = np.array(all_window_features)
                batch_features = torch.tensor(batch_features).permute(0, 3, 1, 2)  # Convert to (N, C, H, W) format
                batch_features = batch_features.to(device)  # Move to device

                feature_batch = extract_features_with_resnet(model, batch_features, device)
                features.extend(feature_batch)

                predictions = adaboost.predict(np.array((features)))
                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        detected_bboxes.append(windows_loc[i])
                                    
            if detected_bboxes:
                detected_bboxes = np.array(detected_bboxes)
                x1 = np.min(detected_bboxes[:, 0])
                y1 = np.min(detected_bboxes[:, 1])
                x2 = np.max(detected_bboxes[:, 2])
                y2 = np.max(detected_bboxes[:, 3])
                all_bboxes.append([x1, y1, x2, y2])
            else:
                all_bboxes.append([])

    return all_bboxes


def resize_bbox(bboxes, original_size, target_size):
    """
    Resize bounding boxes to match the original image dimensions.
    Args:
        bboxes (list of tuples): List of bounding boxes [(x1, y1, x2, y2), ...].
        original_size (tuple): Original size of the image (H, W).
        target_size (tuple): Target size of the image (H', W').
    Returns:
        resized_bboxes (list of tuples): Resized bounding boxes.
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    scale_w, scale_h = orig_w / target_w, orig_h / target_h

    resized_bboxes = []
    x1, y1, x2, y2 = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    resized_bboxes.append((
        math.floor(x1 * scale_w),
        math.floor(y1 * scale_h),
        math.floor(x2 * scale_w),
        math.floor(y2 * scale_h)
    ))
    return resized_bboxes


def main():
    
    parser = get_parser()
    args = parser.parse_args()

    train_loader, val_loader, files_name = load_dataset(args.base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model = model.to(device)
    n_features = 1000
    
    X_train, y_train = prepare_data_for_adaboost(train_loader, model, device)
    clf = AdaBoostClassifier(n_estimators=3, input_dim=n_features, learning_rate=0.01)
    clf.fit(X_train, y_train)

    # Load the best model and test on the validation set
    checkpoint = torch.load("best_adaboost_model.pth")
    
    best_accuracy = checkpoint['accuracy']
    print(f"Best model accuracy: {best_accuracy:.4f}")

    # X_val, y_val = prepare_data_for_adaboost(val_loader, model, device)
    # y_pred = clf.predict(X_val)

    # val_accuracy = np.mean(y_pred == y_val)
    # print(f"Validation accuracy: {val_accuracy:.4f}")

    bboxes = predict_bounding_boxes(val_loader, model, clf, device)
    bbox_link_filename = {}
    os.makedirs(f"{args.output}/bbox", exist_ok=True)
    for i, (image_, bbox) in enumerate(zip(val_loader, bboxes)):
        bbox_link_filename[files_name[i]] = bbox
        if not bbox:
            print(f"Test Image {i}: No bounding box detected.")

    test_path = f"{args.base_dir}/test/image"
    for img_path in os.listdir(test_path):
        img_name = img_path.split(".")[0]
        print(bbox_link_filename[img_name])
        output_txt_path = f"{args.output}/bbox/{img_name}.txt"

        image = cv.imread(os.path.join(test_path, img_path))
        height, width, channels = image.shape
        if bbox_link_filename[img_name]:
            final_bboxes = resize_bbox(bbox_link_filename[img_name], [384, 384], [height, width])
            final_bbox = final_bboxes[0]
            with open(output_txt_path, "w") as txt_file:
                txt_file.write(f"object {final_bbox[0]} {final_bbox[1]} {final_bbox[2]} {final_bbox[3]}\n")

            file_path = f"{args.output}/{img_name}.png"
            visualize_bbox(image, file_path, final_bbox, title=f"Test Image {img_name} - Detected Bounding Box")


if __name__ == "__main__":
    main()
