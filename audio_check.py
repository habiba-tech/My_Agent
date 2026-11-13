
import sounddevice as sd
import numpy as np

for rate in [16000, 44100, 48000]:

    print(f"\nTesting sample rate: {rate} Hz")
    print("Speak normally for 5 seconds...")

    audio = sd.rec(
        int(5 * rate),
        samplerate=rate,
        channels=1,
        dtype="int16",
        device=1
    )

    sd.wait()

    audio = audio.astype(np.float32)

    volume = np.sqrt(np.mean(audio ** 2))
    maximum = np.max(np.abs(audio))

    print("Audio volume:", round(volume, 2))
    print("Maximum sound level:", round(maximum, 2))
