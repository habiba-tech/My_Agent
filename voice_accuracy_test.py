
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# Load offline speech model
print("Loading offline speech model...")
model = Model("vosk-model-small-en-us-0.15")

# Custom command vocabulary
grammar = [
    "hey",
    "turn on the lights",
    "turn off the lights",
    "turn on the fan",
    "turn off the fan",
    "turn on the television",
    "turn off the television",
    "turn on the air conditioner",
    "turn off the air conditioner",
    "open chrome",
    "open notepad",
    "open calculator",
    "what time is it",
    "hello",
    "[unk]"
]

recognizer = KaldiRecognizer(
    model,
    16000,
    json.dumps(grammar)
)

print("\nVoice test started!")
print("Speak a sentence, then pause.")
print("Press Ctrl + C to stop.\n")

try:
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        device=1,
        dtype="int16",
        channels=1
    ) as stream:

        while True:
            data, overflowed = stream.read(4000)

            if recognizer.AcceptWaveform(bytes(data)):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")

                if text:
                    print("Recognized:", text)

except KeyboardInterrupt:
    print("\nVoice test stopped.")

except Exception as error:
    print("Error:", error)
