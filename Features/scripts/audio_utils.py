"""
audio_utils.py
--------------
Loading raw audio files.
"""

import librosa
from config import SR


def load_audio(file_path, sr=SR):
    """Load an audio file, resampled to sr, mono."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y
