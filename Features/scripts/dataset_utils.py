"""
dataset_utils.py
----------------
Finding audio files in the dataset and resolving each one's raga label,
plus a diagnostics function to sanity-check the uuid <-> raga mapping
before you spend time running a full extraction.
"""

import os
from config import AUDIO_EXTS
from uuid_utils import clean_uuid, build_normalized_lookup


def run_diagnostics(root_dir, id_to_raga):
    """Print how well the folder uuids and the id_to_raga dict line up. Run this first."""
    normalized_dict = build_normalized_lookup(id_to_raga)
    folder_uuids = {clean_uuid(f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))}
    dict_uuids = set(normalized_dict.keys())

    missing_from_dict = folder_uuids - dict_uuids
    unused_in_folders = dict_uuids - folder_uuids

    print(f"Folders found: {len(folder_uuids)}")
    print(f"Dict entries: {len(dict_uuids)}")
    print(f"Folders NOT in dict: {len(missing_from_dict)}")
    print(f"Dict keys NOT in folders (expected, e.g. Carnatic-only ragas): {len(unused_in_folders)}")

    if missing_from_dict:
        print("Sample unmatched folder uuids:", list(missing_from_dict)[:10])

    return missing_from_dict


def find_audio_files_with_labels(root_dir, id_to_raga):
    """
    root_dir contains <uuid> subfolders (possibly with duplicate suffixes).
    Returns a list of (file_path, raga_label) for every audio file found.
    """
    normalized_dict = build_normalized_lookup(id_to_raga)
    items = []

    for uuid_folder in sorted(os.listdir(root_dir)):
        uuid_path = os.path.join(root_dir, uuid_folder)
        if not os.path.isdir(uuid_path):
            continue

        raga_label = normalized_dict.get(clean_uuid(uuid_folder))
        if raga_label is None:
            continue  # this uuid isn't in our mapping, skip it

        for dirpath, _, filenames in os.walk(uuid_path):
            for f in filenames:
                if f.lower().endswith(AUDIO_EXTS):
                    items.append((os.path.join(dirpath, f), raga_label))

    return items
