
import sounddevice as sd
import numpy as np

print("Speak loudly for 5 seconds...")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="int16",
    device=1
)

sd.wait()

audio = audio.astype(np.float32)

volume = np.sqrt(np.mean(audio ** 2))

print("Audio volume:", volume)
print("Maximum sound level:", np.max(np.abs(audio)))
