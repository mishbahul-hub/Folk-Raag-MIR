"""
features.py
-----------
The three feature-computation functions, plus a registry so you can
look one up by name (used by the batch extractor).
"""

import numpy as np
import librosa
from gammatone import gtgram

from config import N_FFT, HOP_LENGTH, SR, N_MELS, GT_WIN_TIME, GT_HOP_TIME, GT_CHANNELS, GT_F_MIN
from audio_utils import load_audio


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
