
import joblib
import datetime
import pyttsx3
import json
import sounddevice as sd
import subprocess
import ast
import operator
import re
import time
import numpy as np

from vosk import Model, KaldiRecognizer


# ==================================
# 1. LOAD AI MODELS
# ==================================

print("Loading trained AI model...")
model = joblib.load("jarvis_model.pkl")

print("Loading offline speech model...")
vosk_model = Model("vosk-model-small-en-us-0.15")

print("JARVIS models loaded successfully!")


# ==================================
# 2. INITIALIZE VOICE
# ==================================

engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)


# ==================================
# 3. CALCULATOR OPERATIONS
# ==================================

operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}


# ==================================
# 4. SMART HOME COMMANDS
# ==================================

smart_home_commands = {
    "light_on": "turn on the lights",
    "light_off": "turn off the lights",
    "ac_on": "turn on the air conditioner",
    "ac_off": "turn off the air conditioner",
    "fan_on": "turn on the fan",
    "fan_off": "turn off the fan",
    "vent_on": "turn on the ventilation system",
    "vent_off": "turn off the ventilation system",
    "heater_on": "turn on the heater",
    "heater_off": "turn off the heater",
    "multiplug_on": "turn on the multiplug",
    "multiplug_off": "turn off the multiplug",
    "tv_on": "turn on the television",
    "tv_off": "turn off the television"
}


# ==================================
# 5. SPEAK FUNCTION
# ==================================

def speak(text):

    print("JARVIS:", text)

    engine.say(text)
    engine.runAndWait()


# ==================================
# 6. AUDIO NORMALIZATION
# ==================================

def improve_audio(data):

    audio_array = np.frombuffer(
        data,
        dtype=np.int16
    )

    audio_float = audio_array.astype(np.float32)

    peak = np.max(np.abs(audio_float))

    if peak > 0:

        audio_float = audio_float * (12000 / peak)

    audio_float = np.clip(
        audio_float,
        -32768,
        32767
    ).astype(np.int16)

    return audio_float.tobytes()


# ==================================
# 7. LISTEN FUNCTION
# ==================================

def listen(mode="command"):

    print("\nListening...")

    recognizer = KaldiRecognizer(
        vosk_model,
        16000
    )

    timeout = 15 if mode == "free" else 10

    start_time = time.monotonic()

    try:

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=4000,
            device=1,
            dtype="int16",
            channels=1
        ) as stream:

            while time.monotonic() - start_time < timeout:

                data, overflowed = stream.read(4000)

                if overflowed:
                    print("Warning: Microphone buffer overflow")

                # Improve microphone audio
                audio_data = improve_audio(bytes(data))

                if recognizer.AcceptWaveform(audio_data):

                    result = json.loads(
                        recognizer.Result()
                    )

                    command = result.get(
                        "text", ""
                    ).strip()

                    if command and command != "[unk]":

                        print("You:", command)

                        return command

            # Get remaining speech
            result = json.loads(
                recognizer.FinalResult()
            )

            command = result.get(
                "text", ""
            ).strip()

            if command and command != "[unk]":

                print("You:", command)

                return command

            print("No clear speech detected.")

            return ""

    except Exception as error:

        print("Microphone error:", error)

        return ""


# ==================================
# 8. SAFE CALCULATOR
# ==================================

def calculate(expression):

    try:

        expression = expression.lower()

        expression = expression.replace(
            "multiplied by", "*"
        )

        expression = expression.replace(
            "divided by", "/"
        )

        expression = expression.replace(
            "plus", "+"
        )

        expression = expression.replace(
            "minus", "-"
        )

        expression = expression.replace(
            "times", "*"
        )

        expression = expression.replace(
            "divide", "/"
        )

        expression = expression.replace(
            "x", "*"
        )

        expression = re.sub(
            r"[^0-9.+*/() -]",
            "",
            expression
        )

        tree = ast.parse(
            expression,
            mode="eval"
        )

        def solve(node):

            if isinstance(node, ast.Expression):
                return solve(node.body)

            if (
                isinstance(node, ast.Constant)
                and type(node.value) in (int, float)
            ):

                return node.value

            if (
                isinstance(node, ast.BinOp)
                and type(node.op) in operations
            ):

                return operations[type(node.op)](
                    solve(node.left),
                    solve(node.right)
                )

            raise ValueError("Invalid calculation")

        return solve(tree)

    except Exception:

        return "Sorry, I cannot calculate that."


# ==================================
# 9. SMART HOME HANDLER
# ==================================

def control_smart_home(intent):

    action = smart_home_commands.get(intent)

    if action:

        speak("Smart home command recognized.")

        speak(
            "I understood your request to " + action + "."
        )

        speak("Device control is not connected yet.")

    else:

        speak("Unknown smart home command.")


# ==================================
# 10. PROCESS COMMANDS
# ==================================

def run_jarvis(command):

    if not command:
        return

    probabilities = model.predict_proba(
        [command]
    )[0]

    best_index = probabilities.argmax()

    prediction = model.classes_[best_index]

    confidence = probabilities[best_index]

    print("\nDetected intent:", prediction)

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

    # Confidence check
    if confidence < 0.40:

        speak(
            "Sorry, I did not understand. Please repeat."
        )

        return

    if prediction == "greeting":

        speak("Hello! How can I help you?")

    elif prediction == "time":

        current_time = datetime.datetime.now().strftime(
            "%I:%M %p"
        )

        speak(
            "The current time is " + current_time
        )

    elif prediction == "note":

        speak("What should I remember?")

        note = listen("free")

        if note:

            with open(
                "notes.txt",
                "a",
                encoding="utf-8"
            ) as file:

                file.write(note + "\n")

            speak("Your note has been saved.")

    elif prediction == "calculator":

        speak("Tell me your calculation.")

        expression = listen("free")

        if expression:

            answer = calculate(expression)

            speak(
                "The answer is " + str(answer)
            )

    elif prediction == "open_notepad":

        subprocess.Popen(["notepad.exe"])

        speak("Opening Notepad.")

    elif prediction == "open_chrome":

        speak("Opening Google Chrome.")

        subprocess.Popen(
            ["cmd", "/c", "start", "", "chrome"]
        )

    elif prediction == "open_calculator":

        subprocess.Popen(["calc.exe"])

        speak("Opening Calculator.")

    elif prediction in smart_home_commands:

        control_smart_home(prediction)

    elif prediction == "google_search":

        speak("Google search is not connected yet.")

    else:

        speak("Sorry, I do not know that command yet.")


# ==================================
# 11. CONTINUOUS JARVIS MODE
# ==================================

def start_jarvis():

    speak(
        "Hello! I am Jarvis. Say Hey to activate me."
    )

    while True:

        command = listen("command")

        if not command:
            continue

        command = command.lower().strip()

        if command in [
            "exit",
            "stop jarvis",
            "goodbye"
        ]:

            speak("Goodbye!")

            break

        if "hey" in command.split():

            speak("Yes, I am listening!")

            while True:

                next_command = listen("command")

                if not next_command:
                    continue

                next_command = next_command.lower().strip()

                if next_command in [
                    "exit",
                    "stop jarvis",
                    "goodbye"
                ]:

                    speak("Goodbye!")

                    return

                if next_command in [
                    "sleep",
                    "go to sleep"
                ]:

                    speak(
                        "Going to sleep. Say Hey when you need me."
                    )

                    break

                run_jarvis(next_command)


# ==================================
# 12. RUN PROGRAM
# ==================================

if __name__ == "__main__":

    try:

        start_jarvis()

    except KeyboardInterrupt:

        print("\nJARVIS stopped.")
