"""
Visualize Semantic Segmentation Samples Script

This script provides functions to visualize samples from a semantic segmentation dataset.
Given an image, its corresponding ground truth, and a model's prediction, the script will 
display the image, the overlap between the image and the ground truth, the overlap between 
the image and the model's prediction, and the prediction mask.

Usage:
1. Define your dataset and model.
2. Use the visualize_random_sample function, passing the dataset, model, and device as arguments.
3. The function will select a random sample, make a prediction using the model, and visualize the results.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

# Define colors and class names
colors = ['#FF0000', '#00FF00', '#214525', '#EF2FAC', '#D4AC20', '#668DD5', '#FF5F15', 
          '#0932DD', '#00FFFF', '#7DC79F', '#683987', '#000000', '#800000']

cmap = ListedColormap(colors)

class_names = [
    "Road", "Sidewalk", "Construction", "Fence", "Pole",
    "Traffic Light", "Traffic Sign", "Nature", "Sky", 
    "Person", "Rider", "Car", "Background"
]

def display_segmentation_results(image: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray, 
                                 is_domain_classification: bool = False) -> None:
    """
    Visualize a sample along with its ground truth and model's prediction.
    Also capable of visualizing domain classification results.

    Parameters:
    - image (np.ndarray): Input image.
    - ground_truth (np.ndarray): Ground truth mask.
    - prediction (np.ndarray): Model's prediction mask.
    - is_domain_classification (bool): If True, visualizes domain classification results.
    """
    # Convert tensors to numpy arrays for visualization
    image = np.clip(image.permute(1, 2, 0).numpy(), 0, 1)
    ground_truth, prediction = ground_truth.numpy(), prediction.numpy()

    # Create masks for ground truth and prediction
    gt_mask, pred_mask = np.zeros((*ground_truth.shape, 3)), np.zeros((*prediction.shape, 3))

    if not is_domain_classification:
        for idx, color in enumerate(colors):
            gt_mask[ground_truth == idx] = color
            pred_mask[prediction == idx] = color

    overlap, overlap_with_pred = 0.5 * image + 0.5 * gt_mask, 0.5 * image + 0.5 * pred_mask

    # Plotting
    fig, ax = plt.subplots(1, 4 if not is_domain_classification else 3, figsize=(20, 5))
    ax[0].imshow(image), ax[0].set_title("Image")
    ax[1].imshow(overlap if not is_domain_classification else pred_mask), ax[1].set_title("Ground Truth" if not is_domain_classification else "Prediction")
    if not is_domain_classification:
        ax[2].imshow(overlap_with_pred), ax[2].set_title("Prediction")
        ax[3].imshow(pred_mask), ax[3].set_title("Prediction Mask")
        patches = [mpatches.Patch(color=color, label=class_name) for color, class_name in zip(colors, class_names)]
        ax[3].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()

def monitor_training_process(dataset: torch.utils.data.Dataset, model: torch.nn.Module, 
                             device: torch.device, is_domain_classification: bool = False) -> None:
    """
    Visualize a random sample from the given dataset using the model.

    Parameters:
    - dataset (torch.utils.data.Dataset): Dataset containing images and masks.
    - model (torch.nn.Module): Model to make predictions.
    - device (torch.device): Device to which model and data should be moved before making a prediction.
    - is_domain_classification (bool): If True, visualizes domain classification results.
    """
    random_sample_idx = np.random.randint(len(dataset))
    image, ground_truth = dataset[random_sample_idx]
    
    image = image.unsqueeze(0).to(device)
    model_output = model(image)
    
    if is_domain_classification:
        prediction = model_output[1]
    else:
        _, prediction = model_output[0].max(1)
    
    display_segmentation_results(image[0].cpu(), ground_truth, prediction[0].cpu(), is_domain_classification)