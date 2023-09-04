"""
Fisheye Transformation using PyTorch

This script provides functionality to apply a fisheye transformation effect to images. 
The transformation is implemented using PyTorch, allowing for GPU acceleration. 
Additionally, it can be integrated with the Albumentations library for augmentation purposes.

Functions:
- fisheye_circular_transform : Applies the fisheye effect to an image given by numpy array.
- fisheye_circular_transform_torch: Applies the fisheye effect to an image and optionally a mask.

Classes:
- FisheyeTransform: An Albumentations-compatible class that can be used in augmentation pipelines.
"""

from typing import Tuple, Optional, Union

import torch
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform



def fisheye_circular_transform(image: np.ndarray, 
                               fov_degree: float = 200, 
                               focal_scale: float = 4.5) -> np.ndarray:
    """
    Apply a fisheye effect to the given image using numpy.
    
    The function transforms input image pixels based on radial distance from the center 
    of the image, giving a fisheye lens effect. This effect is achieved using a 
    mathematical transformation in polar coordinates.
    
    Parameters:
    - image (numpy.ndarray): Input image with shape (H, W, C).
    - fov_degree (float): Field of view in degrees. Default is 200.
    - focal_scale (float): Scaling factor for focal length. Default is 4.5.
    
    Returns:
    - numpy.ndarray: Transformed image with fisheye effect.
    """
    
    # Get the shape of the image
    h, w, _ = image.shape
    
    # Calculate the focal length from the given field of view (FOV)
    f = w / (2 * np.tan(0.5 * np.radians(fov_degree)))
    f_scaled = f * focal_scale
    
    # Define the new image
    new_image = np.zeros_like(image)
    
    # Center of the image
    cx, cy = w // 2, h // 2
    
    # Maximum allowable radius (image should fit within this)
    max_radius = cx
    
    for x in range(w):
        for y in range(h):
            # Convert (x, y) to polar coordinates with respect to center
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Check if point is within allowable circle
            if r <= max_radius:
                r_fisheye = f_scaled * np.arctan(r / f_scaled)
                
                # Convert back to Cartesian coordinates
                x_fisheye = int(cx + r_fisheye * np.cos(theta))
                y_fisheye = int(cy + r_fisheye * np.sin(theta))
                
                # Check bounds and assign pixel value
                if 0 <= x_fisheye < w and 0 <= y_fisheye < h:
                    new_image[y, x] = image[y_fisheye, x_fisheye]
                
    return new_image


def fisheye_circular_transform_torch(image: torch.Tensor, 
                                     mask: Optional[torch.Tensor] = None, 
                                     fov_degree: float = 200, 
                                     focal_scale: float = 4.5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply a fisheye effect to the given image and mask using PyTorch.
    
    Parameters:
    - image (torch.Tensor): Input image tensor with shape (C, H, W).
    - mask (torch.Tensor, optional): Input mask tensor. Default is None.
    - fov_degree (float): Field of view in degrees. Default is 200.
    - focal_scale (float): Scaling factor for focal length. Default is 4.5.
    
    Returns:
    - new_image (torch.Tensor): Transformed image with fisheye effect.
    - new_mask (torch.Tensor or None): Transformed mask with fisheye effect or None if no mask is provided.
    """
    _, h, w = image.shape
    radian_conversion = torch.tensor(np.pi/180, dtype=image.dtype, device=image.device)
    f = w / (2 * torch.tan(0.5 * fov_degree * radian_conversion))
    f_scaled = f * focal_scale
    x = torch.linspace(-w//2, w//2, w).repeat(h, 1)
    y = torch.linspace(-h//2, h//2, h).unsqueeze(1).repeat(1, w)
    r = torch.sqrt(x*x + y*y)
    theta = torch.atan2(y, x)
    r_fisheye = f_scaled * torch.atan(r / f_scaled)
    x_fisheye = (w // 2 + r_fisheye * torch.cos(theta)).long()
    y_fisheye = (h // 2 + r_fisheye * torch.sin(theta)).long()
    valid_coords = (x_fisheye >= 0) & (x_fisheye < w) & (y_fisheye >= 0) & (y_fisheye < h) & (r < w // 2) # 렌즈 모양 없는 문제 발견, mask backgroud 0→12로 변경 필요
    new_image = torch.zeros_like(image)
    if mask is not None:
        new_mask = torch.zeros_like(mask)
    else:
        new_mask = None
    new_image[:, valid_coords] = image[:, y_fisheye[valid_coords], x_fisheye[valid_coords]]
    if mask is not None:
        new_mask[:, valid_coords] = mask[:, y_fisheye[valid_coords], x_fisheye[valid_coords]]
    return new_image, new_mask


class FisheyeTransform(ImageOnlyTransform):
    def __init__(self, 
                 fov_degree: float = 200, 
                 focal_scale: float = 4.5, 
                 always_apply: bool = False, 
                 p: float = 1.0):
        """
        Constructor for the FisheyeTransform class.
        
        Parameters:
        - fov_degree (float): Field of view in degrees. Default is 200.
        - focal_scale (float): Scaling factor for focal length. Default is 4.5.
        - always_apply (bool): Flag indicating if the transform should always be applied. Default is False.
        - p (float): Probability of applying the transform. Default is 1.0.
        """
        super(FisheyeTransform, self).__init__(always_apply, p)
        self.fov_degree = fov_degree
        self.focal_scale = focal_scale

    def apply(self, image: Union[np.ndarray, torch.Tensor], **params) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the fisheye transformation to the provided image.
        
        Parameters:
        - image (numpy.ndarray or torch.Tensor): Input image to be transformed.
        
        Returns:
        - Transformed image with fisheye effect.
        """
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        transformed_image, _ = fisheye_circular_transform_torch(image_tensor, fov_degree=self.fov_degree, focal_scale=self.focal_scale)
        return transformed_image.permute(1, 2, 0).byte().numpy()

    def apply_to_mask(self, mask: Union[np.ndarray, torch.Tensor], **params) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the fisheye transformation to the provided mask.
        
        Parameters:
        - mask (numpy.ndarray or torch.Tensor): Input mask to be transformed.
        
        Returns:
        - Transformed mask with fisheye effect.
        """
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()
        _, transformed_mask = fisheye_circular_transform_torch(mask_tensor, fov_degree=self.fov_degree, focal_scale=self.focal_scale)
        return transformed_mask.squeeze(0).byte().numpy()