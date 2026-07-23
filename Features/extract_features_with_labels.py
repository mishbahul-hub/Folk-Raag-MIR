"""
Feature extraction for HMD dataset with raga-label mapping.

Expected folder structure:
    root_dir/
        <uuid>/
            <Singer_Name>/
                <Raag_Folder>/
                    song.wav / song.mp3

id_to_raga: dict mapping uuid (folder name) -> raga name (e.g. from Image 1)

Output: 3 pickle files, each a list of (feature_array, raga_label, file_path)
    - spectrogram_features.pkl
    - logmel_features.pkl
    - cochleagram_features.pkl
"""

import os
import pickle
import numpy as np
import librosa
from gammatone import gtgram
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config (according to Thesis)
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
GT_WIN_TIME = 0.025
GT_HOP_TIME = 0.010
GT_CHANNELS = 128
GT_F_MIN = 50

AUDIO_EXTS = (".mp3", ".wav")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(file_path, sr=SR):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------
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


def extract_all_features(file_path):
    y = load_audio(file_path)
    return {
        "spectrogram": compute_spectrogram(y),
        "logmel": compute_logmel_spectrogram(y),
        "cochleagram": compute_cochleagram(y),
    }


# ---------------------------------------------------------------------------
# Find all audio files + resolve their raga label via the top-level uuid
# ---------------------------------------------------------------------------
def find_audio_files_with_labels(root_dir, id_to_raga):
    """
    root_dir is the 'audio' folder containing <uuid> subfolders.
    Returns list of (file_path, raga_label).
    """
    items = []
    for uuid_folder in sorted(os.listdir(root_dir)):
        uuid_path = os.path.join(root_dir, uuid_folder)
        if not os.path.isdir(uuid_path):
            continue

        raga_label = id_to_raga.get(uuid_folder)
        if raga_label is None:
            continue  # skip uuids not present in the mapping

        for dirpath, _, filenames in os.walk(uuid_path):
            for f in filenames:
                if f.lower().endswith(AUDIO_EXTS):
                    items.append((os.path.join(dirpath, f), raga_label))

    return items


# ---------------------------------------------------------------------------
# Batch processing -> 3 pickle files
# ---------------------------------------------------------------------------
def process_dataset(root_dir, id_to_raga, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    audio_items = find_audio_files_with_labels(root_dir, id_to_raga)

    spectrogram_data = []
    logmel_data = []
    cochleagram_data = []

    for file_path, raga_label in tqdm(audio_items, desc="Extracting features"):
        features = extract_all_features(file_path)

        spectrogram_data.append((features["spectrogram"], raga_label, file_path))
        logmel_data.append((features["logmel"], raga_label, file_path))
        cochleagram_data.append((features["cochleagram"], raga_label, file_path))

    with open(os.path.join(output_dir, "spectrogram_features.pkl"), "wb") as f:
        pickle.dump(spectrogram_data, f)

    with open(os.path.join(output_dir, "logmel_features.pkl"), "wb") as f:
        pickle.dump(logmel_data, f)

    with open(os.path.join(output_dir, "cochleagram_features.pkl"), "wb") as f:
        pickle.dump(cochleagram_data, f)

    print(f"Saved {len(spectrogram_data)} songs -> 3 pickle files in {output_dir}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "/kaggle/input/datasets/mishbhaul/indian-raga-dataset/RagaDataset/Hindustani/audio"  
    OUTPUT_DIR = "/kaggle/working/features"


    id_to_raga = {
    'db085e26-665d-4f4c-a2a5-95251c22b69e': 'Pūrvīkaḷyāṇi',
    '2165542c-45da-4301-af82-c2e7ddbe4768': 'Bilahari',
    '123b09bd-9901-4e64-a65a-10b02c9e0597': 'Bhairavi',
    'df85a0a5-b1a8-42f1-a87b-3d7c7ee33fb4': 'Kēdāragauḷa',
    'cda9cbe9-c1aa-42bb-8b4c-e9dfc8af133c': 'Nāṭakurinji',
    '839bb6b4-1577-4a78-9e5d-4906c6453274': 'Suraṭi',
    'a9413dff-91d1-4e29-ad92-c04019dce5b8': 'Tōḍi',
    '5ce23030-f71d-4f5d-9d76-f91c2c182392': 'Dhanyāsi',
    '978f4c3c-6a12-43fd-8427-c414ee17b252': 'Mukhāri',
    'a4ec6633-1050-4207-8313-b017194c8fa0': 'Kamās',
    'e8a0bf54-13c6-4a09-922a-bfc744ddf38a': 'Varāḷi',
    'c6b5f8d9-ebb4-46af-a020-6646fce2c77d': 'Sāma',
    '993d6cf6-dc89-4d23-9a9f-8eeed1524872': 'Sahānā',
    'e18fcaa7-1f9c-4f09-b627-0687481f4ec7': 'Ānandabhairavi',
    '9cedca68-4a9d-4170-bec3-0d1db1ff730e': 'Māyāmāḷavagauḷa',
    'bdd80890-44f1-4a93-8d93-06a418781f97': 'Śrī',
    'bf4662d4-25c3-4cad-9249-4de2dc513c06': 'Kalyāṇi',
    '50bd048f-4482-4c5b-850c-9ad5e5ec46f1': 'Hussēnī',
    '9a071e54-3eed-48b2-83a3-1a3fd683b8e0': 'Kānaḍa',
    '42dd0ccb-f92a-4622-ae5d-a3be571b4939': 'Śrīranjani',
    '6345e8fe-7061-4bdc-842c-dcfd4a379820': 'Aṭāna',
    '98d46d9e-5100-4f24-bcd4-d0f95966c7cb': 'Sindhubhairavi',
    '18b1acb9-dff6-47ec-873a-b2086c8d268d': 'Madhyamāvati',
    'defad4cc-48aa-4372-a31a-c43624930713': 'Dēvagāndhāri',
    'f0866e71-33b2-47db-ab75-0808c41f2401': 'Kāṁbhōji',
    'a47e9d22-847a-46e8-b589-2a3537789f5f': 'Sāvēri',
    '09c179f3-8b19-4792-a852-e9fa0090e409': 'Kāpi',
    '85ccf631-4cdf-4f6c-a841-0edfcf4255d1': 'Kāmavardani',
    'aa5f376f-06cd-4a69-9cc9-b1077104e0b0': 'Bēgaḍa',
    'e5c4d94a-b34a-42ef-acd7-f235612350e4': 'Gauḷa',
    'a2f9f182-0ceb-4531-b286-b840b47a54b8': 'Śankarābharaṇaṁ',
    '4ce0b18d-f4df-41cf-9b40-9166199506b4': 'Rītigauḷa',
    '700e1d92-7094-4c21-8a3b-d7b60c550edf': 'Behāg',
    '9ebeb536-30a7-403f-8042-f7c1445c4b87': 'Nāṭa',
    '39821826-3327-41d7-9cd5-e22fe7b08360': 'Mōhanaṁ',
    '0277eae5-3411-4b22-9fa8-1b347e7528d1': 'Ṣanmukhapriya',
    'b08475a2-1049-433a-be8f-105bf14718fb': 'Harikāmbhōji',
    'ddff55ae-20f6-4d7d-ba9e-a6c10eeebd41': 'Yadukula kāṁbōji',
    'f972db4d-5d16-4f9a-9841-f313e1601aaa': 'Karaharapriya',
    '3af5a361-923a-465d-864d-9c7ba0c04a47': 'Sencuruṭṭi'
}

    process_dataset(ROOT_DIR, id_to_raga, OUTPUT_DIR)