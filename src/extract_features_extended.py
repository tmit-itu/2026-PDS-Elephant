import pandas as pd
import numpy as np
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from hair_removal import remove_hair
from feature_asymmetry import mean_asymmetry
from feature_compactness import get_compactness
from feature_color import extract_color_features
from feature_texture import mean_gradient
from preprocessing import enhance_color_hsv_clahe


imgs_path = "../data/imgs"
masks_path = "../data/masks"
output_path = "../data/features_extended.csv"


def process_image(row):
    """
    Extract features for a single image using the extended pipeline.

    In addition to baseline features, this version includes hair removal
    and computes hair coverage as an additional feature.

    Parameters:
        row (dict): metadata row containing image information

    Returns:
        dict or None: extracted features or None if processing fails
    """

    try:
        img_id = os.path.splitext(row["img_id"])[0]

        # Load image and corresponding mask
        img = cv2.imread(os.path.join(imgs_path, f"{img_id}.png"))
        mask = cv2.imread(os.path.join(masks_path, f"{img_id}_mask.png"), 0)

        if img is None or mask is None:
            raise ValueError("Image or mask not found")
        
        # Ensure mask matches image size
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Remove hair and compute hair coverage
        img_clean, hair_mask = remove_hair(img)
        h_coverage = np.sum(hair_mask > 0) / hair_mask.size if hair_mask is not None else 0

        # Apply preprocessing (CLAHE)
        img_preprocessed = enhance_color_hsv_clahe(img_clean)

        # Compute features
        feature_a = mean_asymmetry(mask)
        feature_b = get_compactness(mask)
        feature_c = extract_color_features(img_preprocessed, mask)
        feature_t = mean_gradient(img_preprocessed, mask)
        
        return {
            "img_id": row["img_id"],
            "patient_id": row["patient_id"],
            "hair_coverage": h_coverage,
            "cancer": row["cancer"],
            "asymmetry": feature_a,
            "border": feature_b,
            "texture": feature_t,
            "h_mean": feature_c[0],
            "s_mean": feature_c[1],
            "v_mean": feature_c[2],
            "h_std": feature_c[3],
            "s_std": feature_c[4],
            "v_std": feature_c[5],
            "color_entropy": feature_c[6]
            }
    
    except Exception as e:
        print(f"Error processing image {row['img_id']}: {e}")
        return None        


def extract_extended():
    """
    Extract features for all images using the extended pipeline.

    Loads metadata, applies hair removal and preprocessing, and computes
    handcrafted features for each image. Results are saved to a CSV file.
    """

    # Load metadata and map diagnosis to binary labels
    metadata = pd.read_csv("../data/metadata.csv")
    diagnosis_mapping = {'MEL': 1, 'BCC': 1, 'SCC': 1, 'NEV': 0, 'ACK': 0, 'SEK': 0}
    metadata["cancer"] = metadata["diagnostic"].map(diagnosis_mapping)

    # Keep only images available in the folder
    imgs_in_folder = set(os.listdir(imgs_path))
    metadata = metadata[metadata["img_id"].isin(imgs_in_folder)]

    rows = metadata.to_dict("records")
    total = len(rows)

    print(f"Starting feature extraction for {total} images")    

    results = []
    processed = 0

    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, row) for row in rows]

        for future in tqdm(as_completed(futures), total=total):
            res = future.result()
            processed += 1

            if res is not None:
                results.append(res)
            
             # Print progress periodically
            if processed % 100 == 0 or processed == total:
                progress_percentage = (processed / total) * 100
                print(f"Progress: {processed}/{total} images processed ({progress_percentage:.1f}%)")

     
    # Save extracted features
    df_features = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_features.to_csv(output_path, index=False)
    print("Feature extraction completed successfully. CSV file saved.")
    print(f"CSV file saved at: {output_path}")

if __name__ == "__main__":
    extract_extended()