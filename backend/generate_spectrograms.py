import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from parse_ravdess import parse_ravdess, EMOTION_LABELS  # reuse your earlier script

# Constants
SAMPLE_RATE = 22050
N_MELS = 256
FIXED_LENGTH = 256  # More frames for more time resolution

def audio_to_logmel_spectrogram(path, duration=3):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=duration)

    # Ensure fixed-length audio (pad if shorter, trim if longer)
    target_length = SAMPLE_RATE * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmax=8000
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

    # Ensure fixed time width (pad/trim to FIXED_LENGTH)
    if log_mel_spec.shape[1] < FIXED_LENGTH:
        pad_width = FIXED_LENGTH - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :FIXED_LENGTH]

    return log_mel_spec

if __name__ == "__main__":
    RAVDESS_ROOT = "/Users/hanyildirim/Downloads/Audio_Speech_Actors_01-24"
    data = parse_ravdess(RAVDESS_ROOT)

    # Test on a few files
    for i, (path, label) in enumerate(data[:5]):
        spec = audio_to_logmel_spectrogram(path)
        print(f"{i+1}. {label.upper()} | shape: {spec.shape} | path: {os.path.basename(path)}")

        # Optional: show spectrogram
        plt.figure(figsize=(6, 3))
        plt.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        plt.title(f"{label.upper()} - {os.path.basename(path)}")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency Bands")
        plt.colorbar()
        plt.tight_layout()
        plt.show()