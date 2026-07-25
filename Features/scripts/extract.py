"""
extract.py
----------
Batch extraction — the function that actually walks over every song,
computes ONE feature type, and saves it to disk immediately (memory-safe).
"""

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import FEATURE_FUNCTIONS, extract_one_feature
from dataset_utils import find_audio_files_with_labels


def process_dataset_single_feature(root_dir, id_to_raga, output_dir, feature_type):
    assert feature_type in FEATURE_FUNCTIONS, f"feature_type must be one of {list(FEATURE_FUNCTIONS)}"

    feat_dir = os.path.join(output_dir, feature_type)
    os.makedirs(feat_dir, exist_ok=True)

    audio_items = find_audio_files_with_labels(root_dir, id_to_raga)
    print(f"Found {len(audio_items)} audio files. Extracting '{feature_type}'")

    manifest = []       # tiny list — just filenames + labels, never the big arrays
    failed_files = []

    for i, (file_path, raga_label) in enumerate(tqdm(audio_items, desc=feature_type, unit="song")):
        try:
            feat = extract_one_feature(file_path, feature_type)
            feat = feat.astype(np.float16)              # smaller file size, still plenty precise

            out_name = f"{i:04d}_{feature_type}.npy"
            np.save(os.path.join(feat_dir, out_name), feat)

            manifest.append({"file": out_name, "label": raga_label, "source": file_path})

            del feat                                      # drop the array from memory now
        except Exception as e:
            failed_files.append((file_path, str(e)))
        finally:
            gc.collect()                                   # force memory cleanup every song

    manifest_path = os.path.join(output_dir, f"{feature_type}_manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"Saved {len(manifest)} songs -> {feat_dir}/")
    print(f"Manifest written -> {manifest_path}")
    if failed_files:
        print(f"{len(failed_files)} files failed to process:")
        for fp, err in failed_files[:10]:
            print(f"  {fp}: {err}")

    return manifest
