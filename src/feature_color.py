
import cv2 
import numpy as np
from scipy.stats import entropy

def mean_hsv_in_mask(img, mask):
    """
    Compute the mean HSV color values within the lesion region.

    Converts the image to HSV and averages pixel values inside the mask.

    Parameters:
        img (ndarray): input image in BGR format
        mask (ndarray): binary mask of the lesion region

    Returns:
        array-like: mean HSV values [H, S, V]
    """

    # Convert to HSV and extract masked pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    pixels = hsv[mask > 0].astype(np.float32) / 255.0

    return np.mean(pixels, axis=0) if len(pixels) > 0 else [0.0, 0.0, 0.0]


def std_hsv_in_mask(img, mask):
    """
    Compute the standard deviation of HSV values within the lesion.

    Measures color variability inside the masked region.

    Parameters:
        img (ndarray): input image in BGR format
        mask (ndarray): binary mask of the lesion region

    Returns:
        array-like: standard deviation of HSV values [H, S, V]
    """

    # Convert to HSV and extract masked pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    pixels = hsv[mask > 0].astype(np.float32) / 255.0

    return np.std(pixels, axis=0) if len(pixels) > 0 else [0.0, 0.0, 0.0]


def color_entropy_from_pixels(img, mask):
    """
    Compute entropy of the hue distribution within the lesion.

    Entropy captures how varied the color distribution is, with higher
    values indicating more complex or heterogeneous coloration.

    Parameters:
        img (ndarray): input image in BGR format
        mask (ndarray): binary mask of the lesion region

    Returns:
        float: entropy of hue values
    """

    # Extract hue values inside the mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask  = mask[:, :, 0]
    
    h_channel = hsv[:, :, 0][mask > 0].astype(np.float32) / 255.0
    
    if len(h_channel) == 0:
        return 0.0
    
    # Compute histogram and entropy
    hist, _ = np.histogram(h_channel, bins = 20, range=(0,1))
    hist_dist = hist / (np.sum(hist) + 1e-6)

    return float(entropy(hist_dist + 1e-6))

def extract_color_features(image, mask):
    """
    Extract combined color features from a lesion.

    Includes mean HSV values, standard deviation of HSV,
    and entropy of hue distribution.

    Parameters:
        image (ndarray): input image in BGR format
        mask (ndarray): binary mask of the lesion region

    Returns:
        ndarray: concatenated color feature vector
    """
    features = []
    
    # Mean color
    features.extend(list(mean_hsv_in_mask(image, mask)))

    # Color variability
    features.extend(list(std_hsv_in_mask(image, mask)))

    # Color distribution complexity
    features.append(color_entropy_from_pixels(image, mask))
        
    return np.array(features, dtype=np.float32)