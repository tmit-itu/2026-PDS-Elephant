import pandas as pd
import cv2
import os
from hair_removal import remove_hair
from feature_asymmetry import mean_asymmetry
from feature_compactness import get_compactness
from feature_texture import mean_gradient
from feature_diameter import get_diameter

def extract_extended():
    metadata = pd.read_csv("../metadata.csv")
    diagnosis_mapping = {'MEL': 1, 'BCC': 1, 'SCC': 1, 'NEV': 0, 'ACK': 0, 'SEK': 0}
    metadata["cancer"] = metadata["diagnostic"].map(diagnosis_mapping)

    results = []

    for index, row in metadata.iterrows():
        img_id = os.path.splitext(row["img_id"])[0]
        img = cv2.imread(f"../data/imgs/{img_id}.png")
        mask = cv2.imread(f"../data/masks/{img_id}_mask.png", 0)

        if img is not None and mask is not None:
            img_clean, _ = remove_hair(img)

            feature_a = mean_asymmetry(mask)
            feature_b = get_border(mask)
            feature_t = mean_gradient(img_clean, mask)
            feature_d = get_diameter(mask)
            
            results.append({
                "img_id": img_id,
                "asymmetry": feature_a,
                "border": feature_b,
                "texture": feature_t,
                "diameter": feature_d,
                "cancer": row["cancer"]
            })
    
    df_features = pd.DataFrame(results)
    df_features.to_csv("../results/features_extended.csv", index=False)

if __name__ == "__main__":
    extract_extended()