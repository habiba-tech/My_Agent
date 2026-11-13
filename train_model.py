
# ==========================================
# JARVIS - COMPLETE UNIFIED AI MODEL TRAINING
# ==========================================

import os
import json
import glob
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ==========================================
# 1. BASIC JARVIS COMMANDS
# ==========================================

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
    "start my calculator",

    # Additional heater commands
    "I'm freezing",
    "im freezing",
    "I am freezing",
    "I'm feeling cold",
    "I need some warmth",
    "It's very cold in here",
    "Turn on the heater",
    "Please switch on the heater",
    "Can you make the room warmer",
    "I want to feel warm",
    "I'm shivering",
    "The room is too cold",
    "Start the heating",
    "I need heating",
    "It's chilly in here",

    # Additional AC commands
    "I'm feeling hot",
    "I am feeling hot",
    "It's too hot in here",
    "The room is very hot",
    "I need some cooling",
    "I'm sweating",
    "Make the room cooler",
    "I can't tolerate this heat",
    "The temperature is too high",
    "I'm feeling overheated"
]


labels = [

    # Greeting
    "greeting", "greeting", "greeting",
    "greeting", "greeting",

    # Time
    "time", "time", "time", "time",

    # Notes
    "note", "note", "note", "note",

    # Calculator
    "calculator", "calculator",
    "calculator", "calculator",

    # Notepad
    "open_notepad", "open_notepad",
    "open_notepad", "open_notepad",

    # Chrome
    "open_chrome", "open_chrome",
    "open_chrome", "open_chrome",
    "open_chrome",

    # Windows Calculator
    "open_calculator", "open_calculator",
    "open_calculator", "open_calculator",
    "open_calculator",

    # Heater commands
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",

    # AC commands
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on"
]


all_data = [
    pd.DataFrame({
        "text": sentences,
        "intent": labels
    })
]


# ==========================================
# 2. LOAD CSV DATASETS
# ==========================================

def load_csv_files(folder):

    files = glob.glob(
        os.path.join(folder, "**", "*.csv"),
        recursive=True
    )

    for file in files:

        try:

            df = pd.read_csv(file)

            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            text_column = next(
                (c for c in [
                    "instruction",
                    "text",
                    "sentence",
                    "command",
                    "utterance",
                    "query",
                    "input"
                ] if c in df.columns),
                None
            )

            intent_column = next(
                (c for c in [
                    "intent",
                    "label",
                    "category",
                    "class"
                ] if c in df.columns),
                None
            )

            if text_column and intent_column:

                part = df[
                    [text_column, intent_column]
                ].copy()

                part.columns = ["text", "intent"]

                all_data.append(part)

                print(
                    "Loaded CSV:",
                    file,
                    "| Examples:",
                    len(part)
                )

            else:

                print(
                    "Skipped CSV:",
                    file
                )

        except Exception as e:

            print(
                "CSV loading error:",
                file,
                e
            )


# ==========================================
# 3. LOAD JSONL DATASETS
# ==========================================

def load_jsonl_files(folder):

    files = glob.glob(
        os.path.join(folder, "**", "*.jsonl"),
        recursive=True
    )

    for file in files:

        rows = []

        try:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    if not line.strip():
                        continue

                    try:

                        item = json.loads(line)

                        if not isinstance(item, dict):
                            continue

                        text = None
                        intent = None

                        # Find command text
                        for key in [
                            "instruction",
                            "text",
                            "sentence",
                            "command",
                            "utterance",
                            "query",
                            "input"
                        ]:

                            if isinstance(item.get(key), str):

                                text = item[key]
                                break

                        # Find intent
                        for key in [
                            "intent",
                            "label",
                            "category",
                            "class"
                        ]:

                            if isinstance(item.get(key), str):

                                intent = item[key]
                                break

                        if text and intent:

                            rows.append({
                                "text": text,
                                "intent": intent
                            })

                    except json.JSONDecodeError:

                        continue

            if rows:

                part = pd.DataFrame(rows)

                all_data.append(part)

            print(
                "Loaded JSONL:",
                file,
                "| Examples:",
                len(rows)
            )

        except Exception as e:

            print(
                "JSONL loading error:",
                file,
                e
            )


# ==========================================
# 4. LOAD ALL DATASETS
# ==========================================

print("\n========== LOADING ALL DATASETS ==========")

load_csv_files("data/raw")
load_jsonl_files("data/raw")

load_csv_files("data/processed")
load_jsonl_files("data/processed")


# ==========================================
# 5. COMBINE ALL DATA
# ==========================================

print("\n========== COMBINING DATA ==========")

data = pd.concat(
    all_data,
    ignore_index=True
)

data["text"] = (
    data["text"]
    .astype("string")
    .str.strip()
)

data["intent"] = (
    data["intent"]
    .astype("string")
    .str.strip()
    .str.lower()
)

data = data.dropna(
    subset=["text", "intent"]
)

data = data[
    (data["text"] != "") &
    (data["intent"] != "")
]

print(
    "Total examples before cleaning:",
    len(data)
)


# ==========================================
# 6. REMOVE CONFLICTING COMMANDS
# ==========================================

print("\n========== CHECKING CONFLICTS ==========")

data["normalized_text"] = (
    data["text"]
    .str.lower()
    .str.strip()
)

conflicts = data.groupby(
    "normalized_text"
)["intent"].nunique()

conflict_texts = conflicts[
    conflicts > 1
].index

if len(conflict_texts) > 0:

    print(
        "Conflicting commands found:",
        len(conflict_texts)
    )

    print(
        data[
            data["normalized_text"].isin(conflict_texts)
        ][["text", "intent"]]
        .drop_duplicates()
        .to_string(index=False)
    )

    data = data[
        ~data["normalized_text"].isin(conflict_texts)
    ]

else:

    print("No conflicting commands found.")


# ==========================================
# 7. REMOVE DUPLICATES
# ==========================================

data = data.drop_duplicates(
    subset=["normalized_text", "intent"]
)

data = data.drop(
    columns=["normalized_text"]
)

print("\n========== FINAL DATASET ==========")

print(
    "Total clean examples:",
    len(data)
)

print(
    "Total unique intents:",
    data["intent"].nunique()
)

print("\nExamples per intent:")

print(
    data["intent"].value_counts().sort_index()
)


# ==========================================
# 8. CHECK DATA BEFORE TRAINING
# ==========================================

if data.empty:

    raise ValueError(
        "No training data found!"
    )

if data["intent"].nunique() < 2:

    raise ValueError(
        "At least two intents are required!"
    )

if data["intent"].value_counts().min() < 2:

    raise ValueError(
        "Every intent needs at least two examples."
    )


# ==========================================
# 9. CREATE ONE UNIFIED AI MODEL
# ==========================================

model = Pipeline([

    (
        "vectorizer",
        CountVectorizer(
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

])


# ==========================================
# 10. TRAIN USING ALL CLEAN DATA
# ==========================================

print("\n========== TRAINING JARVIS ==========")

model.fit(
    data["text"].tolist(),
    data["intent"].tolist()
)

print("Training completed successfully!")


# ==========================================
# 11. SAVE ONE JARVIS MODEL
# ==========================================

joblib.dump(
    model,
    "jarvis_model.pkl"
)

print("\nJARVIS model saved successfully!")

print("Model file: jarvis_model.pkl")

print(
    "Total training examples:",
    len(data)
)

print(
    "Total trained intents:",
    data["intent"].nunique()
)


# ==========================================
# 12. TEST TRAINED MODEL
# ==========================================

print("\n========== JARVIS AI TEST ==========")

print("Type 'exit' to stop testing.")

while True:

    command = input("\nYou: ").strip()

    if command.lower() == "exit":

        print("JARVIS testing finished!")

        break

    if not command:
        continue

    prediction = model.predict([command])

    probabilities = model.predict_proba([command])[0]

    confidence = max(probabilities)

    print(
        "JARVIS understood:",
        prediction[0]
    )

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

# ==========================================
# JARVIS - COMPLETE UNIFIED AI MODEL TRAINING
# ==========================================

import os
import json
import glob
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ==========================================
# 1. BASIC JARVIS COMMANDS
# ==========================================

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
    "start my calculator",

    # Additional heater commands
    "I'm freezing",
    "im freezing",
    "I am freezing",
    "I'm feeling cold",
    "I need some warmth",
    "It's very cold in here",
    "Turn on the heater",
    "Please switch on the heater",
    "Can you make the room warmer",
    "I want to feel warm",
    "I'm shivering",
    "The room is too cold",
    "Start the heating",
    "I need heating",
    "It's chilly in here",

    # Additional AC commands
    "I'm feeling hot",
    "I am feeling hot",
    "It's too hot in here",
    "The room is very hot",
    "I need some cooling",
    "I'm sweating",
    "Make the room cooler",
    "I can't tolerate this heat",
    "The temperature is too high",
    "I'm feeling overheated",

    # More natural AC commands
    "Please cool this room down",
    "Can you cool the room?",
    "Make it cooler in here",
    "The room feels too warm",
    "I need the room to be colder",
    "Reduce the room temperature",
    "Turn on the cooling",
    "It's getting hot inside",
    "I need the AC because it's hot",
    "Please lower the temperature"
]


# ==========================================
# 2. MATCHING INTENT LABELS
# ==========================================

labels = [

    # Greeting
    "greeting", "greeting", "greeting",
    "greeting", "greeting",

    # Time
    "time", "time", "time", "time",

    # Notes
    "note", "note", "note", "note",

    # Calculator
    "calculator", "calculator",
    "calculator", "calculator",

    # Notepad
    "open_notepad", "open_notepad",
    "open_notepad", "open_notepad",

    # Chrome
    "open_chrome", "open_chrome",
    "open_chrome", "open_chrome",
    "open_chrome",

    # Windows Calculator
    "open_calculator", "open_calculator",
    "open_calculator", "open_calculator",
    "open_calculator",

    # Heater commands
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",
    "heater_on",

    # Original AC commands
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",

    # More natural AC labels
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on",
    "ac_on"
]


# ==========================================
# 3. CHECK BASIC DATA
# ==========================================

if len(sentences) != len(labels):
    raise ValueError(
        f"Sentence/label mismatch: "
        f"{len(sentences)} sentences and "
        f"{len(labels)} labels"
    )


all_data = [
    pd.DataFrame({
        "text": sentences,
        "intent": labels
    })
]


# ==========================================
# 4. LOAD CSV DATASETS
# ==========================================

def load_csv_files(folder):

    files = glob.glob(
        os.path.join(folder, "**", "*.csv"),
        recursive=True
    )

    for file in files:

        try:

            df = pd.read_csv(file)

            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            text_column = next(
                (c for c in [
                    "instruction",
                    "text",
                    "sentence",
                    "command",
                    "utterance",
                    "query",
                    "input"
                ] if c in df.columns),
                None
            )

            intent_column = next(
                (c for c in [
                    "intent",
                    "label",
                    "category",
                    "class"
                ] if c in df.columns),
                None
            )

            if text_column and intent_column:

                part = df[
                    [text_column, intent_column]
                ].copy()

                part.columns = ["text", "intent"]

                all_data.append(part)

                print(
                    "Loaded CSV:",
                    file,
                    "| Examples:",
                    len(part)
                )

            else:

                print(
                    "Skipped CSV:",
                    file
                )

        except Exception as e:

            print(
                "CSV loading error:",
                file,
                e
            )


# ==========================================
# 5. LOAD JSONL DATASETS
# ==========================================

def load_jsonl_files(folder):

    files = glob.glob(
        os.path.join(folder, "**", "*.jsonl"),
        recursive=True
    )

    for file in files:

        rows = []

        try:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    if not line.strip():
                        continue

                    try:

                        item = json.loads(line)

                        if not isinstance(item, dict):
                            continue

                        text = None
                        intent = None

                        # Find command text
                        for key in [
                            "instruction",
                            "text",
                            "sentence",
                            "command",
                            "utterance",
                            "query",
                            "input"
                        ]:

                            if isinstance(item.get(key), str):

                                text = item[key]
                                break

                        # Find intent
                        for key in [
                            "intent",
                            "label",
                            "category",
                            "class"
                        ]:

                            if isinstance(item.get(key), str):

                                intent = item[key]
                                break

                        if text and intent:

                            rows.append({
                                "text": text,
                                "intent": intent
                            })

                    except json.JSONDecodeError:

                        continue

            if rows:

                part = pd.DataFrame(rows)

                all_data.append(part)

            print(
                "Loaded JSONL:",
                file,
                "| Examples:",
                len(rows)
            )

        except Exception as e:

            print(
                "JSONL loading error:",
                file,
                e
            )


# ==========================================
# 6. LOAD ALL DATASETS
# ==========================================

print("\n========== LOADING ALL DATASETS ==========")

load_csv_files("data/raw")
load_jsonl_files("data/raw")

load_csv_files("data/processed")
load_jsonl_files("data/processed")


# ==========================================
# 7. COMBINE ALL DATA
# ==========================================

print("\n========== COMBINING DATA ==========")

data = pd.concat(
    all_data,
    ignore_index=True
)

data["text"] = (
    data["text"]
    .astype("string")
    .str.strip()
)

data["intent"] = (
    data["intent"]
    .astype("string")
    .str.strip()
    .str.lower()
)

data = data.dropna(
    subset=["text", "intent"]
)

data = data[
    (data["text"] != "") &
    (data["intent"] != "")
]

print(
    "Total examples before cleaning:",
    len(data)
)


# ==========================================
# 8. REMOVE CONFLICTING COMMANDS
# ==========================================

print("\n========== CHECKING CONFLICTS ==========")

data["normalized_text"] = (
    data["text"]
    .str.lower()
    .str.strip()
)

conflicts = data.groupby(
    "normalized_text"
)["intent"].nunique()

conflict_texts = conflicts[
    conflicts > 1
].index

if len(conflict_texts) > 0:

    print(
        "Conflicting commands found:",
        len(conflict_texts)
    )

    print(
        data[
            data["normalized_text"].isin(conflict_texts)
        ][["text", "intent"]]
        .drop_duplicates()
        .to_string(index=False)
    )

    data = data[
        ~data["normalized_text"].isin(conflict_texts)
    ]

else:

    print("No conflicting commands found.")


# ==========================================
# 9. REMOVE DUPLICATES
# ==========================================

data = data.drop_duplicates(
    subset=["normalized_text", "intent"]
)

data = data.drop(
    columns=["normalized_text"]
)

print("\n========== FINAL DATASET ==========")

print(
    "Total clean examples:",
    len(data)
)

print(
    "Total unique intents:",
    data["intent"].nunique()
)

print("\nExamples per intent:")

print(
    data["intent"].value_counts().sort_index()
)


# ==========================================
# 10. CHECK DATA BEFORE TRAINING
# ==========================================

if data.empty:

    raise ValueError(
        "No training data found!"
    )

if data["intent"].nunique() < 2:

    raise ValueError(
        "At least two intents are required!"
    )

if data["intent"].value_counts().min() < 2:

    raise ValueError(
        "Every intent needs at least two examples."
    )


# ==========================================
# 11. CREATE ONE UNIFIED AI MODEL
# ==========================================

model = Pipeline([

    (
        "vectorizer",
        CountVectorizer(
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

])


# ==========================================
# 12. TRAIN USING ALL CLEAN DATA
# ==========================================

print("\n========== TRAINING JARVIS ==========")

model.fit(
    data["text"].tolist(),
    data["intent"].tolist()
)

print("Training completed successfully!")


# ==========================================
# 13. SAVE ONE JARVIS MODEL
# ==========================================

joblib.dump(
    model,
    "jarvis_model.pkl"
)

print("\nJARVIS model saved successfully!")

print("Model file: jarvis_model.pkl")

print(
    "Total training examples:",
    len(data)
)

print(
    "Total trained intents:",
    data["intent"].nunique()
)


# ==========================================
# 14. TEST TRAINED MODEL
# ==========================================

print("\n========== JARVIS AI TEST ==========")

print("Type 'exit' to stop testing.")

while True:

    command = input("\nYou: ").strip()

    if command.lower() == "exit":

        print("JARVIS testing finished!")

        break

    if not command:
        continue

    prediction = model.predict([command])

    probabilities = model.predict_proba([command])[0]

    confidence = max(probabilities)

    print(
        "JARVIS understood:",
        prediction[0]
    )

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )
