import matplotlib.pyplot as plt
import librosa.display

def plot_features(feats, sr=16000, hop_length=160):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Spectrogram
    librosa.display.specshow(feats["spectrogram"], sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="log", ax=axes[0])
    axes[0].set_title("Spectrogram")

    # 2. log Mel-spectrogram
    librosa.display.specshow(feats["logmel"], sr=sr, hop_length=hop_length,
                              x_axis="time", y_axis="log", ax=axes[1])
    axes[1].set_title("Log Mel-spectrogram")

    #3. Cochleagram
    librosa.display.specshow(feats["Cochleagram"], sr=sr, hop_length=hop_length,
                              x_axis="time", y_axis="log", ax=axes[2])
    axes[2].set_title("Cochleagram")

    for ax in axes:
        fig.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.show()