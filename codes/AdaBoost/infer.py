import torch
import numpy as np
from joblib import load

def infer_single_image(image, model, adaboost_model, device, window_size=(32, 32), stride=16, threshold=0.5):
    """
    Perform inference on a single image using AdaBoost and extracted features.

    Args:
        image: Input image tensor of shape (C, H, W).
        model: Pre-trained feature extraction model (e.g., ResNet).
        adaboost_model: Trained AdaBoost classifier.
        device: PyTorch device (e.g., 'cuda' or 'cpu').
        window_size: Tuple specifying the size of the sliding window (height, width).
        stride: Stride for sliding the window.
        threshold: Threshold for determining foreground labels.

    Returns:
        dict: A dictionary containing window predictions and their coordinates.
    """
    model.eval()  # Set the model to evaluation mode

    # Convert image to numpy for sliding window
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    h, w = img_np.shape[:2]

    all_window_features = []
    window_coordinates = []

    # Sliding window
    for y in range(0, h - window_size[1] + 1, stride):
        for x in range(0, w - window_size[0] + 1, stride):
            window = img_np[y:y + window_size[1], x:x + window_size[0]]
            window_coordinates.append((x, y))
            all_window_features.append(window)

    if len(all_window_features) > 0:
        # Convert windows to tensor
        batch_features = np.array(all_window_features)
        batch_features = torch.tensor(batch_features).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        batch_features = batch_features.to(device)

        # Extract features
        with torch.no_grad():
            feature_batch = model(batch_features).cpu().numpy()
        print(feature_batch.shape)
        # Predict with AdaBoost
        window_predictions = adaboost_model.predict(feature_batch)

        return {
            "window_predictions": window_predictions,
            "window_coordinates": window_coordinates
        }

    return {"window_predictions": [], "window_coordinates": []}


# Example usage
if __name__ == "__main__":
    from torchvision.models import resnet18
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    import augmentation

    # Load the pre-trained model and AdaBoost model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet18(pretrained=True)
    resnet_model.to(device)

    adaboost_model_path = "output/model.joblib"
    adaboost_model = load(adaboost_model_path)

    # Load and preprocess an image
    image_path = "../../output_results/original_image.png"
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize to fixed dimensions for consistency
        transforms.ToTensor()
    ])
    image_tensor = transform(image).to(device)

    # Perform inference
    results = infer_single_image(image_tensor, resnet_model, adaboost_model, device)

    # Display predictions
    print("Predicted windows and coordinates:")
    for prediction, coord in zip(results["window_predictions"], results["window_coordinates"]):
        print(f"Window at {coord} - Prediction: {prediction}")

    # Optional: visualize the predictions
    plt.imshow(image)
    for coord, prediction in zip(results["window_coordinates"], results["window_predictions"]):
        x, y = coord
        if prediction == 1:  # Highlight predicted foreground regions
            plt.gca().add_patch(plt.Rectangle((x, y), 32, 32, edgecolor='red', facecolor='none', linewidth=2))
    plt.savefig("output.png")
