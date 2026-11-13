
import pandas as pd
import os

# ==========================================
# BASIC JARVIS COMMAND DATASET
# ==========================================

commands = {

    "greeting": [
        "hello",
        "hi",
        "hey",
        "hey jarvis",
        "hello jarvis",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you doing",
        "are you there",
        "wake up jarvis",
        "nice to meet you",
        "what's up",
        "hey buddy",
        "hello assistant",
        "good to see you",
        "how is it going",
        "are you ready",
        "hi buddy",
        "greetings",
        "hey friend",
        "hello there",
        "jarvis are you awake",
        "start listening jarvis",
        "can you hear me",
        "how are things",
        "let's begin",
        "good to have you back"
    ],

    "time": [
        "what time is it",
        "tell me the time",
        "show me the current time",
        "current time please",
        "can you tell me the time",
        "what is the time now",
        "check the time",
        "display the time",
        "do you know the time",
        "tell me what time it is",
        "what's the time",
        "give me the current time",
        "I want to know the time",
        "please check the clock",
        "show current time",
        "jarvis tell me the time",
        "can you check the clock",
        "what time do we have",
        "tell me the clock time",
        "I need to know the time",
        "time please",
        "check current time",
        "show me the clock",
        "what is the present time",
        "read the current time",
        "give me the time",
        "jarvis what time is it",
        "tell me the present time",
        "can you show the time",
        "check what time it is"
    ],

    "note": [
        "save a note",
        "remember this",
        "remember this for me",
        "write a note",
        "save this information",
        "take a note",
        "make a note",
        "store this note",
        "remember my assignment",
        "save my reminder",
        "write this down",
        "keep this information",
        "note this down",
        "I want to save something",
        "please remember this",
        "store my information",
        "create a new note",
        "add a note",
        "save my notes",
        "remember my task",
        "write my reminder",
        "keep a note",
        "save this for later",
        "record this information",
        "jarvis take a note",
        "I have something to remember",
        "please write this down",
        "store this for me",
        "remember my work",
        "make a quick note"
    ],

    "calculator": [
        "calculate 20 plus 30",
        "what is 10 times 5",
        "add 50 and 60",
        "calculate 100 divided by 2",
        "what is 25 plus 75",
        "subtract 20 from 100",
        "multiply 12 by 5",
        "divide 100 by 10",
        "calculate 45 plus 55",
        "what is 9 times 9",
        "solve 50 minus 20",
        "calculate 200 divided by 4",
        "what is 15 multiplied by 3",
        "add 100 and 200",
        "subtract 30 from 90",
        "what is 80 divided by 8",
        "calculate 25 times 4",
        "solve 150 plus 250",
        "what is 500 minus 125",
        "multiply 20 by 6",
        "calculate 75 divided by 5",
        "what is 40 plus 60",
        "solve 12 times 12",
        "add 250 and 150",
        "calculate 300 minus 75",
        "what is 144 divided by 12",
        "multiply 15 by 8",
        "calculate 90 plus 110",
        "what is 1000 minus 500",
        "solve 25 multiplied by 10"
    ],

    "open_notepad": [
        "open notepad",
        "start notepad",
        "launch notepad",
        "open my notepad",
        "please open notepad",
        "start the notepad application",
        "launch the notepad app",
        "I want to open notepad",
        "can you open notepad",
        "run notepad",
        "open windows notepad",
        "start my notepad",
        "bring up notepad",
        "show me notepad",
        "open the text editor",
        "run the notepad application",
        "jarvis open notepad",
        "I need notepad",
        "please start notepad",
        "can you launch notepad",
        "open a text editor",
        "start the text editor",
        "I want to use notepad",
        "open notepad for me",
        "launch windows notepad",
        "start writing in notepad",
        "get notepad ready",
        "run the text editor",
        "please launch notepad",
        "open the notepad program"
    ],

    "open_chrome": [
        "open chrome",
        "start chrome",
        "launch google chrome",
        "open google browser",
        "start google browser",
        "please open chrome",
        "run chrome",
        "launch chrome browser",
        "open my browser",
        "start my browser",
        "I want to open chrome",
        "can you launch chrome",
        "open the internet browser",
        "start the chrome application",
        "bring up google chrome",
        "jarvis open chrome",
        "I need google chrome",
        "please start the browser",
        "launch my web browser",
        "open chrome for me",
        "run google chrome",
        "start browsing",
        "show me google chrome",
        "open the google browser",
        "get chrome ready",
        "I want to use chrome",
        "launch the browser",
        "please launch google chrome",
        "open my google browser",
        "start the chrome browser"
    ],

    "open_calculator": [
        "open calculator",
        "start calculator",
        "launch calculator",
        "open windows calculator",
        "start my calculator",
        "please open calculator",
        "run calculator",
        "launch the calculator app",
        "open my calculator",
        "I need the calculator",
        "can you open calculator",
        "start the calculator application",
        "bring up calculator",
        "show me calculator",
        "jarvis open calculator",
        "I want to use calculator",
        "please start calculator",
        "run windows calculator",
        "open calculator for me",
        "launch windows calculator",
        "get calculator ready",
        "start my windows calculator",
        "open the calculation app",
        "I want to calculate something",
        "please launch calculator",
        "can you start the calculator",
        "open the calculator program",
        "show windows calculator",
        "start the calculation tool",
        "run the calculator application"
    ]
}


# ==========================================
# CONVERT TO DATAFRAME
# ==========================================

rows = []

for intent, phrases in commands.items():

    for phrase in phrases:

        rows.append({
            "text": phrase,
            "intent": intent
        })


df = pd.DataFrame(rows)

# Remove duplicate commands
df["normalized"] = (
    df["text"].str.lower().str.strip()
)

df = df.drop_duplicates(
    subset=["normalized", "intent"]
)

df = df.drop(columns=["normalized"])


# ==========================================
# SAVE DATASET
# ==========================================

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/jarvis_basic_commands.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n========== BASIC DATASET CREATED ==========")

print("Total examples:", len(df))

print("\nExamples per intent:")

print(df["intent"].value_counts().sort_index())

print("\nSaved successfully:", output_file)
