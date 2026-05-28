import cv2

def remove_hair(img):
    """
    Remove hair artifacts from a skin lesion image using morphological filtering
    and inpainting.

    The method detects hair using a blackhat transformation, creates a binary
    mask of hair pixels, and then removes them using inpainting.

    Parameters:
        img (ndarray): input image in BGR format

    Returns:
        dst (ndarray): image with hair removed
        hair_mask (ndarray): binary mask indicating detected hair regions
    """

    # Convert to grayscale and detect dark hair structures using blackhat
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to obtain binary hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Remove hair using inpainting guided by the mask
    dst = cv2.inpaint(img, hair_mask, inpaintRadius = 1, flags = cv2.INPAINT_TELEA)

    return dst, hair_mask