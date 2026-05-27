
import cv2 
import numpy as np
from scipy.stats import entropy

def mean_hsv_in_mask(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    pixels = hsv[mask > 0].astype(np.float32) / 255.0

    if len(pixels) > 0:
        return np.mean(pixels, axis=0)
    
    return [0.0, 0.0, 0.0]

def std_hsv_in_mask(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    pixels = hsv[mask > 0].astype(np.float32) / 255.0

    if len(pixels) > 0:
        return np.std(pixels, axis=0)
        
    return [0.0, 0.0, 0.0]

def color_entropy_from_pixels(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if len(mask.shape) == 3:
        mask  = mask[:, :, 0]
    h_channel = hsv[:, :, 0][mask > 0].astype(np.float32) / 255.0
    if len(h_channel) == 0: return 0.0
    hist, _ = np.histogram(h_channel, bins = 20, range=(0,1))
    hist_dist = hist / (np.sum(hist) + 1e-6)
    return float(entropy(hist_dist + 1e-6))

def extract_color_features(image, mask):
    
    """Extract color features from an object in the image.
    The features describe color variability, dominant colors 
    and color asymmetry inside the masked object.
    """
    features = []
    
    mean_hsv = mean_hsv_in_mask(image, mask)
    features.extend(list(mean_hsv))   

    std_hsv = std_hsv_in_mask(image, mask)
    features.extend(list(std_hsv))   

    features.append(color_entropy_from_pixels(image, mask))
        
    return np.array(features, dtype=np.float32)
