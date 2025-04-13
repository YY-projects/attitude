import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

from parse_ravdess import parse_ravdess, EMOTION_LABELS
from generate_spectrograms import audio_to_logmel_spectrogram

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.data = torch.tensor(np.array(spectrograms)).unsqueeze(1).float()
        self.labels = torch.tensor(labels).long()  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def prepare_dataset(root_dir, emotion_labels=EMOTION_LABELS):
    filepaths = parse_ravdess(root_dir)
    print(f"🎧 Total files: {len(filepaths)}")

    spectrograms = []
    raw_labels = []

    for path, label in filepaths:
        try:
            spec = audio_to_logmel_spectrogram(path)
            spectrograms.append(spec)
            raw_labels.append(label)
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)

    print("✅ Encoded labels:", list(label_encoder.classes_))

    # Split into training and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        spectrograms, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )

    train_dataset = SpectrogramDataset(X_train, y_train)
    val_dataset = SpectrogramDataset(X_val, y_val)
    print(f"Shape of first spectrogram: {train_dataset[0][0].shape}")

    return train_dataset, val_dataset, label_encoder

if __name__ == "__main__":
    root = "/Users/hanyildirim/Downloads/Audio_Speech_Actors_01-24"
    train_dataset, val_dataset, label_encoder = prepare_dataset(root)

    print(f"🧠 Train: {len(train_dataset)} samples")
    print(f"🧪 Val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for x, y in train_loader:
        print(f"🔢 Batch shape: {x.shape} Labels: {y[:5]}")
        break