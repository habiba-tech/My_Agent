
import sounddevice as sd

for rate in [16000, 22050, 44100, 48000]:
    try:
        sd.check_input_settings(
            device=1,
            samplerate=rate,
            channels=1,
            dtype="int16"
        )
        print(rate, "Hz: Supported")

    except Exception as e:
        print(rate, "Hz: Not supported")
