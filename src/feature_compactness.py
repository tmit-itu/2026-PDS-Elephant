import numpy as np
from skimage import measure


def get_compactness(mask):
    """
    Compute the compactness of a lesion from its binary mask.

    Compactness measures how closely the shape resembles a circle.
    A value close to 1 indicates a circular shape, while higher values
    indicate more irregular or complex boundaries.

    Parameters:
        mask (ndarray): binary mask of the lesion region

    Returns:
        float: compactness value
    """

    # Ensure mask is binary and single-channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    binary_mask = mask > 0
    
    # Compute lesion area
    area = np.sum(binary_mask)
    if area == 0:
        return 0.0 
    
    # Compute perimeter of the lesion
    perimeter = measure.perimeter(binary_mask)
    if perimeter == 0:
        return 0.0
    
    # Apply compactness formula
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    return float(compactness)