
# JARVIS - Personal AI Assistant

import joblib
import datetime
import subprocess
import ast
import operator

# Load our trained AI model
model = joblib.load("jarvis_model.pkl")

# Safe calculator operations
operations = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}

def calculate(expression):
    try:
        tree = ast.parse(expression, mode="eval")

        def solve(node):
            if isinstance(node, ast.Expression):
                return solve(node.body)

            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                left = solve(node.left)
                right = solve(node.right)
                return operations[type(node.op)](left, right)

            raise ValueError("Invalid calculation")

        return solve(tree)

    except Exception:
        return "Sorry, invalid calculation."


# Start Jarvis
print("================================")
print("       JARVIS AI ASSISTANT")
print("================================")
print("Jarvis is ready!")

while True:

    command = input("\nYou: ")

    if command.lower() == "exit":
        print("Jarvis: Goodbye!")
        break

    # AI predicts the command
    prediction = model.predict([command])[0]

    # Perform actions
    if prediction == "greeting":
        print("Jarvis: Hello! How can I help you?")

    elif prediction == "time":
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        print("Jarvis: The current time is", current_time)

    elif prediction == "note":
        note = input("Jarvis: What should I remember? ")
        with open("notes.txt", "a", encoding="utf-8") as file:
            file.write(note + "\n")
        print("Jarvis: Your note has been saved!")

    elif prediction == "calculator":
        expression = input("Jarvis: Enter calculation (example: 20 + 30): ")
        answer = calculate(expression)
        print("Jarvis: Answer =", answer)

    elif prediction == "open_notepad":
        subprocess.Popen(["notepad.exe"])
        print("Jarvis: Opening Notepad!")

    else:
        print("Jarvis: Sorry, I don't understand.")
