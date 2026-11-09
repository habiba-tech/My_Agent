
# JARVIS - AI Model Training

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Training sentences
sentences = [

    # Greeting
    "hello",
    "hi jarvis",
    "good morning",
    "hey there",
    "how are you",

    # Time
    "what time is it",
    "tell me the time",
    "show current time",
    "current time please",

    # Notes
    "remember my assignment",
    "save a note",
    "write this note",
    "remember this for me",

    # Calculator
    "calculate 20 plus 30",
    "what is 10 times 5",
    "add 50 and 60",
    "calculate 100 divided by 2",

    # Notepad
    "open notepad",
    "start notepad",
    "launch notepad",
    "open my notepad",

    # Chrome
    "open chrome",
    "start chrome",
    "launch google chrome",
    "open google browser",
    "start google browser",

    # Windows Calculator
    "open calculator",
    "start calculator",
    "launch calculator",
    "open windows calculator",
    "start my calculator"
]

# Training labels
labels = [

    # Greeting
    "greeting",
    "greeting",
    "greeting",
    "greeting",
    "greeting",

    # Time
    "time",
    "time",
    "time",
    "time",

    # Notes
    "note",
    "note",
    "note",
    "note",

    # Calculator
    "calculator",
    "calculator",
    "calculator",
    "calculator",

    # Notepad
    "open_notepad",
    "open_notepad",
    "open_notepad",
    "open_notepad",

    # Chrome
    "open_chrome",
    "open_chrome",
    "open_chrome",
    "open_chrome",
    "open_chrome",

    # Windows Calculator
    "open_calculator",
    "open_calculator",
    "open_calculator",
    "open_calculator",
    "open_calculator"
]

# Create AI model
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(sentences, labels)

# Save model
joblib.dump(model, "jarvis_model.pkl")

print("JARVIS AI model trained successfully!")

# Test model
while True:

    command = input("\nYou: ")

    if command.lower() == "exit":
        print("Training test finished!")
        break

    prediction = model.predict([command])

    print("JARVIS understood:", prediction[0])
