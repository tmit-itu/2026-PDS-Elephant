import numpy as np
from math import floor, ceil
from skimage.transform import rotate


'''Calculating asymmetry score by rotating image n times over 180 degrees,
cropping after each rotation, and calculating asymmetry each time,
with the final score taken as the average of all rotations.'''

def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask
    '''

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    coords = np.argwhere(mask > 0)
    
    if coords.size == 0:
        return mask

    mins = coords.min(axis = 0)
    maxs = coords.max(axis=0)

    
    row_min, col_min = mins[0], mins[1]
    row_max, col_max = maxs[0], maxs[1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

def find_midpoint(image):
    '''Find midpoint of image array.'''
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def asymmetry(mask):
    ''' Calculate asymmetry score between 0 and 1 from vertical and horizontal axis
        on a binary mask, 0 being complete symmetry, 1 being complete asymmetry,
       i.e. no pixels overlapping when folding mask on x- and y-axis

       Args:
        mask (numpy.ndarray): input mask

       Returns:
          asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.
    '''

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
    '''Rotate mask n times and calculate asymmetry score for each iteration.
    Rotates n times between 0 and 180 degrees.

    Args:
        mask (numpy.ndarray): Input binary mask.
        n (int): Number of rotations.

    Returns:
        dict: Asymmetry scores for each rotation angle.
    '''
    asymmetry_scores = {}
    mask = mask.astype(bool)

    for i in range(n):

        degrees = 180 * i / n

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
    '''Return mean asymmetry score from mask.
    Chose 4 instead of the defaulted 30 so that the machine runs faster. 

    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (default 4)

    Returns:
        mean_score (float): mean asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score
