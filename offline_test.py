
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

print("Loading Vosk model...")
model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

print("Vosk is ready!")
print("\nSpeak clearly for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="int16",
    device=1
)

sd.wait()

print("\nProcessing audio...")

for i in range(0, len(audio), 4000):
    chunk = audio[i:i + 4000].tobytes()

    if recognizer.AcceptWaveform(chunk):
        result = json.loads(recognizer.Result())
        print("Recognized:", result.get("text", ""))

final_result = json.loads(recognizer.FinalResult())

print("\nFinal Vosk result:")
print(final_result.get("text", ""))
