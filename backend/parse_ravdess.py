import os
from pathlib import Path

RAVDESS_ROOT = Path("/Users/hanyildirim/Downloads/Audio_Speech_Actors_01-24")

# Emotion codes
EMOTION_LABELS = {
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "02": "happy",
    "06": "sad",
    "07": "angry"
}

def parse_ravdess(root: Path):
    data = []

    for subdir, _, files in os.walk(root):
        for filename in files:
            if filename.endswith(".wav"):
                parts = filename.split("-")
                if len(parts) < 4:
                    continue  # not a valid filename

                emotion_code = parts[2]
                if emotion_code in EMOTION_LABELS:
                    full_path = Path(subdir) / filename
                    label = EMOTION_LABELS[emotion_code]
                    data.append((str(full_path), label))

    return data

if __name__ == "__main__":
    data = parse_ravdess(RAVDESS_ROOT)
    print(f"✅ Found {len(data)} matching audio files.")
    for path, label in data[:10]:  # preview first 10
        print(f"{label:6} → {path}")