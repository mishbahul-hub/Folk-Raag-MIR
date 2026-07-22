"""
Feature extraction for Hindustani Music Dataset (HMD) - Raga Identification
Computes: Spectrogram, Log-Mel Spectrogram, Cochleagram

Expected folder structure:
    root_dir/
        Raga1/
            song1.mp3
            song2.mp3
        Raga2/
            song1.mp3
        ...

Install (Kaggle):
    !pip install librosa gammatone -q
"""

import os
import numpy as np
import librosa
from gammatone import gtgram
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SR = 16000          # sample rate to resample audio to
N_FFT = 1024         # FFT window size
HOP_LENGTH = 160     # hop size (10 ms at 16kHz)
N_MELS = 64          # number of mel bands
GT_WIN_TIME = 0.025  # gammatone window (25 ms)
GT_HOP_TIME = 0.010  # gammatone hop (10 ms)
GT_CHANNELS = 64     # number of gammatone filters
GT_F_MIN = 50        # lowest center frequency (Hz)


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
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db


def compute_logmel_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def compute_cochleagram(y, sr=SR, win_time=GT_WIN_TIME, hop_time=GT_HOP_TIME,
                         channels=GT_CHANNELS, f_min=GT_F_MIN):
    coch = gtgram.gtgram(y, sr, win_time, hop_time, channels, f_min)
    coch_db = librosa.power_to_db(coch, ref=np.max)
    return coch_db


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------
def extract_all_features(file_path):
    y = load_audio(file_path)
    features = {
        "spectrogram": compute_spectrogram(y),
        "logmel": compute_logmel_spectrogram(y),
        "cochleagram": compute_cochleagram(y),
    }
    return features


# ---------------------------------------------------------------------------
# Batch processing over the dataset
# ---------------------------------------------------------------------------
def process_dataset(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_exts = (".mp3", ".wav")

    for raga_name in sorted(os.listdir(root_dir)):
        raga_path = os.path.join(root_dir, raga_name)
        if not os.path.isdir(raga_path):
            continue

        out_raga_dir = os.path.join(output_dir, raga_name)
        os.makedirs(out_raga_dir, exist_ok=True)

        song_files = [f for f in os.listdir(raga_path) if f.lower().endswith(audio_exts)]

        for song_file in tqdm(song_files, desc=raga_name):
            file_path = os.path.join(raga_path, song_file)
            song_name = os.path.splitext(song_file)[0]

            features = extract_all_features(file_path)

            for feat_name, feat_array in features.items():
                save_path = os.path.join(out_raga_dir, f"{song_name}_{feat_name}.npy")
                np.save(save_path, feat_array)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "/kaggle/input/hmd-dataset"   # change to your dataset path
    OUTPUT_DIR = "/kaggle/working/features"   # change to your desired output path

    process_dataset(ROOT_DIR, OUTPUT_DIR)
