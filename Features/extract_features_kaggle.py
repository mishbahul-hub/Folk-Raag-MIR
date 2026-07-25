"""
extract_features_kaggle.py
---------------------------
HMD Raga Feature Extraction Script — memory-safe, one feature at a time.

WHAT THIS SCRIPT DOES
1. Finds every audio file in the HMD dataset and figures out which raga it belongs to
   (using the uuid folder name + a raga lookup dictionary).
2. Computes ONE of: spectrogram / log-mel spectrogram / cochleagram, for every song.
3. Saves each song's feature as its own small .npy file (instead of one giant file),
   so memory usage stays low and stable no matter how many songs you have.
4. Writes a small manifest.csv that maps each saved .npy file -> its raga label.

HOW TO RUN IT ON KAGGLE
    !python extract_features_kaggle.py --feature spectrogram
    !python extract_features_kaggle.py --feature logmel
    !python extract_features_kaggle.py --feature cochleagram

Run it three times (once per feature) — each run only computes one feature type,
which keeps memory usage low and lets you stop/restart between runs safely.
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import librosa
from gammatone import gtgram
from tqdm import tqdm


# ===========================================================================
# 1. CONFIG — all the tunable settings live here, in one place
# ===========================================================================
SR = 16000            # sample rate every audio file gets resampled to
N_FFT = 512            # FFT window size
HOP_LENGTH = 320        # hop size between frames
N_MELS = 64             # number of mel bands (for log-mel spectrogram)
GT_WIN_TIME = 0.025     # gammatone window length in seconds (for cochleagram)
GT_HOP_TIME = 0.010     # gammatone hop length in seconds
GT_CHANNELS = 128       # number of gammatone filters (for cochleagram)
GT_F_MIN = 50           # lowest frequency the gammatone filterbank covers (Hz)

AUDIO_EXTS = (".mp3", ".wav")
UUID_LEN = 36            # a uuid is always 36 characters long

ROOT_DIR = "/kaggle/input/datasets/mishbhaul/indian-raga-dataset/RagaDataset/Hindustani/audio"
OUTPUT_DIR = "/kaggle/working/features_with_labels"

# uuid (dataset folder name) -> raga name, for every raga we have audio for
ID_TO_RAGA = {
    'f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8 2': 'Rāgēśrī',
    'f6432fec-e9c2-4b09-9e73-c46086cbd8ea': 'Ahira bhairav',
    '290674e0-d94c-41c1-ad99-f65fa22a1447': 'Madhuvanti',
    '0b3bbf97-0ec3-41da-add4-722d87329ec3': 'Madhukauns',
    'a9ee554f-f146-43e9-b2d0-9bb31fd4b57d': 'Kēdār',
    'd9c603fa-875f-4b84-b851-c6a345427898': 'Ābhōgī',
    '1e7de02f-e77f-405a-a033-f31117aaf955': 'Bhairav',
    '3eb7ba30-4b94-432e-9618-875ee57e01ab': 'Mārvā',
    '46997b02-f09c-4969-8138-4e1861f61967': 'Śrī',
    '54c4214c-05b9-4acc-8f77-6d5786e43a2e': 'Mārūbihāg',
    'a99e07d5-20a0-467b-8dcd-aa5a095177fd': 'Lalit',
    '62b79291-73b0-4c77-a353-0f8bc6ed8362': 'Dēś',
    '118401e7-8de8-4d81-9d8e-8070889e3fa8': 'Darbāri',
    '7591faad-e68a-4550-b675-8082842c6056': 'Bāgēśrī',
    '93c73081-bdf8-4eca-b325-d736b71e9b4b': 'Khamāj',
    'ba3242d0-8d40-4d93-865e-d2e2497ea2a8': 'Gauḍ malhār',
    'ecd04ceb-b46c-47fc-9045-84ac9160e527': 'Basant',
    '6f13484e-6fdd-402d-baf3-3835835454d0': 'Tōḍī',
    'fa28470c-d413-44c7-94da-181f530cbfdd': 'Puriyā dhanaśrī',
    'f7fddfc0-8c1d-4dd2-90d5-5d51a99d61f8': 'Rāgēśrī',
    '40dbe1db-858f-4366-bef8-9076eb67340d': 'Śuddh sāraṅg',
    '1b05a564-059f-445b-b325-cf26318367e3': 'Miyān malhār',
    'b143adaa-f1a6-4de4-8985-a5bd35e96279': 'Bairāgi',
    '64e5fb9e-5569-4e80-8e6c-f543af9469c7': 'Mālkauns',
    '2ed9379f-14c9-49af-8e4d-f9b63e96801f': 'Alahaiyā bilāval',
    'e771a74d-545d-41d5-816b-43403a818b0c': 'Bhūp',
    '063ea5a0-23b1-4bb5-8537-3d924fe8ebb3': 'Jōg',
    'a7d98897-e2fe-4d75-b9dc-e5b4dc88e1c6': 'Bihāg',
    '48b37bed-e847-4882-8a01-5c721e07f07d': 'Yaman kalyāṇ',
    '595de771-fcc2-414c-9df6-ae4ffd898549': 'Hamsadhvāni',
    'dd59147d-8775-44ff-a36b-0d9f15b31319': 'Bilāsakhānī tōḍī',
}


# ===========================================================================
# 2. AUDIO LOADING
# ===========================================================================
def load_audio(file_path, sr=SR):
    """Load an audio file, resampled to sr, mono."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y


# ===========================================================================
# 3. FEATURE FUNCTIONS — one function per feature type
# ===========================================================================
def compute_spectrogram(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    return librosa.amplitude_to_db(S, ref=np.max)


def compute_logmel_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def compute_cochleagram(y, sr=SR, win_time=GT_WIN_TIME, hop_time=GT_HOP_TIME,
                         channels=GT_CHANNELS, f_min=GT_F_MIN):
    coch = gtgram.gtgram(y, sr, win_time, hop_time, channels, f_min)
    return librosa.power_to_db(coch, ref=np.max)


# a lookup table: feature name (string) -> the function that computes it
FEATURE_FUNCTIONS = {
    "spectrogram": compute_spectrogram,
    "logmel": compute_logmel_spectrogram,
    "cochleagram": compute_cochleagram,
}


def extract_one_feature(file_path, feature_type):
    """Load one audio file and compute only the requested feature."""
    y = load_audio(file_path)
    compute_fn = FEATURE_FUNCTIONS[feature_type]
    return compute_fn(y)


# ===========================================================================
# 4. ROBUST UUID MATCHING
#    Kaggle sometimes renames duplicate folders to "uuid 2", "uuid 3", etc,
#    and casing can differ — these two helpers make matching resilient to that.
# ===========================================================================
def clean_uuid(folder_name):
    """Strip any duplicate-folder suffix (e.g. 'uuid 2' -> 'uuid') and normalize case."""
    return folder_name[:UUID_LEN].strip().lower()


def build_normalized_lookup(id_to_raga):
    """Lowercased uuid -> raga_name lookup, so matching is case-insensitive."""
    return {k.strip().lower(): v for k, v in id_to_raga.items()}


# ===========================================================================
# 5. DIAGNOSTICS — run this before extracting anything, to catch mapping bugs early
# ===========================================================================
def run_diagnostics(root_dir, id_to_raga):
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


# ===========================================================================
# 6. FIND EVERY AUDIO FILE + ITS RAGA LABEL
# ===========================================================================
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


# ===========================================================================
# 7. BATCH EXTRACTION — memory-safe: saves each song immediately, one at a time
# ===========================================================================
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


# ===========================================================================
# 8. ENTRY POINT
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract one HMD raga feature type at a time.")
    parser.add_argument(
        "--feature",
        choices=list(FEATURE_FUNCTIONS.keys()),
        required=True,
        help="Which feature to extract: spectrogram, logmel, or cochleagram",
    )
    parser.add_argument("--root_dir", default=ROOT_DIR, help="Path to the folder containing <uuid> subfolders")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Where to save the extracted features")
    args = parser.parse_args()

    print("Running diagnostics first...")
    run_diagnostics(args.root_dir, ID_TO_RAGA)

    print()
    process_dataset_single_feature(
        root_dir=args.root_dir,
        id_to_raga=ID_TO_RAGA,
        output_dir=args.output_dir,
        feature_type=args.feature,
    )


if __name__ == "__main__":
    main()
