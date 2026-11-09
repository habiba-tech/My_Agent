
import sounddevice as sd
import numpy as np

print("JARVIS microphone test")
print("Speak something for 5 seconds...")

# Record audio for 5 seconds
audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="int16"
)

sd.wait()

print("Recording completed!")

if np.max(np.abs(audio)) > 0:
    print("Microphone is receiving sound!")
else:
    print("No sound detected. Check your microphone.")
