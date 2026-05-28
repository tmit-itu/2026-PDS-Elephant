import numpy as np
from math import floor, ceil
from skimage.transform import rotate

def cut_mask(mask):
    """
    Crop mask to the smallest region containing the lesion.

    Removes empty space around the lesion to ensure accurate symmetry
    calculations.

    Parameters:
        mask (ndarray): binary mask

    Returns:
        ndarray: cropped mask
    """

    # Ensure mask is single-channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return mask

    # Determine bounding box and crop
    mins = coords.min(axis = 0)
    maxs = coords.max(axis=0)

    row_min, col_min = mins[0], mins[1]
    row_max, col_max = maxs[0], maxs[1]

    return mask[row_min:row_max+1, col_min:col_max+1]


def find_midpoint(image):
    """
    Compute the geometric midpoint of the image.
    """
    return image.shape[0] / 2, image.shape[1] / 2


def asymmetry(mask):
    """
    Compute asymmetry score based on horizontal and vertical axes.

    The mask is split into halves and compared using XOR to measure
    pixel differences. The score ranges from 0 (symmetric) to 1 (asymmetric).

    Parameters:
        mask (ndarray): binary mask

    Returns:
        float: asymmetry score
    """

    row_mid, col_mid = find_midpoint(mask)

    # Split mask into halves hortizontally and vertically
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Flip one half for each axis
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # Use logical xor to find pixels where only one half is present
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    # Compute sums of total pixels and pixels in asymmetry areas
    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    # Calculate asymmetry score
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)


def rotation_asymmetry(mask, n: int):
    """
    Compute asymmetry scores across multiple rotations.

    The mask is rotated between 0° and 180°, and asymmetry is evaluated
    for each orientation.

    Parameters:
        mask (ndarray): binary mask
        n (int): number of rotations

    Returns:
        dict: mapping of rotation angle -> asymmetry score
    """

    asymmetry_scores = {}
    mask = mask.astype(bool)

    for i in range(n):
        degrees = 180 * i / n

        # Rotate mask, threshold to binary, and crop
        rotated_mask = rotated_mask = rotate(
            mask.astype(float),
            degrees,
            resize=True,
            order=0,
            preserve_range=True
        ) > 0.5

        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores


def mean_asymmetry(mask, rotations = 4):
    """
    Compute mean asymmetry over multiple rotations.

    A small number of rotations (default = 4) is used for efficiency.

    Parameters:
        mask (ndarray): binary mask
        rotations (int): number of rotations

    Returns:
        float: mean asymmetry score
    """
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)

    return sum(asymmetry_scores.values()) / len(asymmetry_scores)
