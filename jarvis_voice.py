
import joblib
import datetime
import pyttsx3
import json
import sounddevice as sd
import subprocess
import ast
import operator

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
# 4. SPEAK FUNCTION
# ==================================

def speak(text):

    print("JARVIS:", text)

    engine.say(text)
    engine.runAndWait()


# ==================================
# 5. OFFLINE LISTEN FUNCTION
# ==================================

def listen():

    print("\nListening...")

    recognizer = KaldiRecognizer(vosk_model, 16000)

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

                    command = result.get("text", "").strip()

                    if command:

                        print("You:", command)

                        return command

    except Exception as error:

        print("Microphone error:", error)

        return ""


# ==================================
# 6. SAFE CALCULATOR
# ==================================

def calculate(expression):

    try:

        expression = expression.lower()

        expression = expression.replace("multiplied by", "*")
        expression = expression.replace("divided by", "/")
        expression = expression.replace("plus", "+")
        expression = expression.replace("minus", "-")
        expression = expression.replace("times", "*")
        expression = expression.replace("divide", "/")

        tree = ast.parse(expression, mode="eval")

        def solve(node):

            if isinstance(node, ast.Expression):
                return solve(node.body)

            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in operations:

                return operations[type(node.op)](
                    solve(node.left),
                    solve(node.right)
                )

            raise ValueError("Invalid calculation")

        return solve(tree)

    except Exception:

        return "Sorry, I cannot calculate that."


# ==================================
# 7. PROCESS COMMANDS
# ==================================

def run_jarvis(command):

    if not command:
        return

    prediction = model.predict([command])[0]

    print("Detected intent:", prediction)

    if prediction == "greeting":

        speak("Hello! How can I help you?")

    elif prediction == "time":

        current_time = datetime.datetime.now().strftime("%I:%M %p")

        speak("The current time is " + current_time)

    elif prediction == "note":

        speak("What should I remember?")

        note = listen()

        if note:

            with open("notes.txt", "a", encoding="utf-8") as file:

                file.write(note + "\n")

            speak("Your note has been saved.")

    elif prediction == "calculator":

        speak("Tell me your calculation.")

        expression = listen()

        answer = calculate(expression)

        speak("The answer is " + str(answer))

    elif prediction == "open_notepad":

        subprocess.Popen(["notepad.exe"])

        speak("Opening Notepad.")

    elif prediction == "open_chrome":

        speak("Opening Google Chrome.")

        subprocess.Popen(["cmd", "/c", "start", "", "chrome"])

    elif prediction == "open_calculator":

        speak("Opening Calculator.")

        subprocess.Popen(["calc.exe"])

    else:

        speak("Sorry, I do not know that command yet.")


# ==================================
# 8. START JARVIS
# ==================================

speak("Hello! I am Jarvis. Say Hey to activate me.")

while True:

    command = listen()

    if not command:
        continue

    command = command.lower().strip()

    # Exit command
    if command in ["exit", "stop jarvis", "goodbye"]:

        speak("Goodbye!")

        break

    # Wake word: HEY
    if "hey" in command.split():

        speak("Yes, I am listening!")

        next_command = listen()

        if not next_command:
            continue

        next_command = next_command.lower().strip()

        if next_command in ["exit", "stop jarvis", "goodbye"]:

            speak("Goodbye!")

            break

        run_jarvis(next_command)
