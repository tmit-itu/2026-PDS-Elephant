import cv2

def enhance_color_hsv_clahe(img, clip_limit = 2.0, tile_grid_size = (8, 8)):
    """
    Enhance image contrast using CLAHE applied to the Value channel in HSV.

    The image is converted to HSV color space, and CLAHE is applied only
    to the Value (brightness) channel. This improves visibility of lesion
    details while preserving color information (Hue and Saturation).

    Parameters:
        img (ndarray): input image in BGR format
        clip_limit (float): CLAHE clip limit controlling contrast strength
        tile_grid_size (tuple): size of grid for CLAHE tiles

    Returns:
        ndarray: contrast-enhanced image in BGR format
    """

    # Convert to HSV and split into channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply CLAHE to the Value (brightness) channel
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_grid_size)
    v_enhanced = clahe.apply(v)

    # Merge channels and convert back to BGR
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced_image