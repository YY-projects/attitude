from proto_out import audio_response_pb2

with open("response.pb", "rb") as f:
    data = f.read()

response = audio_response_pb2.AudioResponse()
response.ParseFromString(data)

print("Transcript:", response.transcript)
print("Emotion:", response.emotion)
print("Confidence:", response.confidence)