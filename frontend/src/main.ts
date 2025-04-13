import protobuf from "protobufjs";

console.log("🎉 Script is RUNNING!");
const recordButton = document.getElementById("recordBtn") as HTMLButtonElement;

let mediaRecorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let isRecording = false;

async function decodeProtobuf(buffer: ArrayBuffer) {
    try {
        const root = await protobuf.load("src/audio_response.proto"); // Relative to index.html
        const AudioResponse = root.lookupType("AudioResponse");

        const message = AudioResponse.decode(new Uint8Array(buffer));
        const object = AudioResponse.toObject(message);

        console.log("✅ Transcript:", object.transcript);
        console.log("✅ Emotion:", object.emotion);

        alert(`Transcript: ${object.transcript}\nEmotion: ${object.emotion}`);
    } catch (err) {
        console.error("❌ Failed to decode Protobuf:", err);
    }
}

recordButton.addEventListener("click", async () => {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            chunks = [];

            mediaRecorder.ondataavailable = (event) => {
                chunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(chunks, { type: "audio/webm" });

                const formData = new FormData();
                formData.append("file", audioBlob, "recording.webm");

                try {
                    const response = await fetch("http://127.0.0.1:8000/upload-audio/", {
                        method: "POST",
                        headers: { "Accept": "application/x-protobuf" },
                        body: formData
                    });

                    if (!response.ok) throw new Error("Upload failed");

                    const buffer = await response.arrayBuffer();
                    await decodeProtobuf(buffer);
                } catch (err) {
                    console.error("❌ Error sending audio:", err);
                }
            };

            mediaRecorder.start();
            isRecording = true;
            recordButton.textContent = "Stop Recording";
            recordButton.classList.add("recording");
        } catch (err) {
            console.error("❌ Microphone access denied:", err);
        }
    } else {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            isRecording = false;
            recordButton.textContent = "Start Recording";
            recordButton.classList.remove("recording");
        }
    }
});