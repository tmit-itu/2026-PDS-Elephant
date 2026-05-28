import cv2
import numpy as np 

def mean_gradient(image,mask):
    """
    Compute a texture feature based on the mean gradient magnitude within the lesion.

    The image is resized for efficiency, converted to grayscale, and Sobel filters
    are used to compute gradient magnitude. The mean gradient is then calculated
    over the lesion area defined by the mask.

    Parameters:
        image (ndarray): input image in BGR format
        mask (ndarray): binary mask of the lesion region

    Returns:
        float: mean gradient magnitude (texture feature)
    """
    # Resize image and mask for faster computation
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Convert to grayscale and compute Sobel gradients
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)

    # Compute gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Ensure mask is single-channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Extract gradient values inside the lesion
    texture_values = magnitude[mask > 0]

    # Handle empty mask case
    if len(texture_values) == 0:
        return 0.0
    
    return float(np.mean(texture_values))