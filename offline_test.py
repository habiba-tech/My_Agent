
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

print("Loading Vosk model...")
model = Model("vosk-model-small-en-us-0.15")

print("Jarvis is ready!")
print("Get ready to speak...")

input("Press ENTER to start recording...")

print("NOW SPEAK: Hello Jarvis")

audio = sd.rec(
    int(6 * 16000),
    samplerate=16000,
    channels=1,
    dtype="int16",
    device=1
)

sd.wait()

recognizer = KaldiRecognizer(model, 16000)
recognizer.SetWords(True)
recognizer.AcceptWaveform(audio.tobytes())

result = json.loads(recognizer.FinalResult())

print("\nYou said:", result["text"])
print("Confidence details:", result.get("result", []))
