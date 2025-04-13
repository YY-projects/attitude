from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import numpy as np
import librosa
from emotion_cnn import EmotionCNN
import subprocess
import tempfile
import os

# 🧠 Protobuf
from proto_out import audio_response_pb2  # Generated using protoc

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔓 Allow frontend from localhost:5173 (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🗣️ Load Whisper model
whisper_model = whisper.load_model("base")

# 😠😢😊 Load Emotion classifier
emotion_model = EmotionCNN(num_classes=3)
emotion_model.load_state_dict(torch.load("emotion_cnn.pt", map_location=torch.device('cpu')))
emotion_model.eval()

emotion_labels = ["angry", "happy", "sad"]
SAMPLE_RATE = 22050
N_MELS = 256
FIXED_LENGTH = 256

def preprocess_audio_for_emotion(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    target_len = SAMPLE_RATE * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())

    if log_mel.shape[1] < FIXED_LENGTH:
        log_mel = np.pad(log_mel, ((0, 0), (0, FIXED_LENGTH - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :FIXED_LENGTH]

    tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    return tensor

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_bytes = await file.read()

    # Save the WebM input to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
        temp_input.write(file_bytes)
        temp_input_path = temp_input.name

    # Convert WebM to WAV
    temp_wav_path = temp_input_path.replace(".webm", ".wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_input_path,
            "-ar", str(SAMPLE_RATE), temp_wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return Response(content=b"ffmpeg failed", status_code=500)

    # Run transcription
    whisper_result = whisper_model.transcribe(temp_wav_path)
    transcript = whisper_result.get("text", "").strip()

    # Predict emotion
    input_tensor = preprocess_audio_for_emotion(temp_wav_path)
    with torch.no_grad():
        prediction = emotion_model(input_tensor)
        predicted_idx = prediction.argmax(dim=1).item()
        emotion = emotion_labels[predicted_idx]

    # Clean up temp files
    os.remove(temp_input_path)
    os.remove(temp_wav_path)

    # Build Protobuf response
    response_proto = audio_response_pb2.AudioResponse(
        transcript=transcript,
        emotion=emotion
    )

    return Response(
        content=response_proto.SerializeToString(),
        media_type="application/x-protobuf"
    )