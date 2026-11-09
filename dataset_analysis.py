
import pandas as pd
import json
from pathlib import Path

folder = Path("data/raw")

all_data = []

# Read CSV files
for file in folder.glob("*.csv"):

    df = pd.read_csv(file)

    df = df[["text", "intent"]]

    df["source"] = file.name

    all_data.append(df)


# Read JSONL files
for file in folder.glob("*.jsonl"):

    rows = []

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            rows.append({
                "text": item.get("instruction", ""),
                "intent": item.get("intent", "")
            })

    df = pd.DataFrame(rows)

    df["source"] = file.name

    all_data.append(df)


# Combine datasets
data = pd.concat(all_data, ignore_index=True)

# Clean text
data["text"] = data["text"].fillna("").str.strip()
data["intent"] = data["intent"].fillna("").str.strip()

data = data[
    (data["text"] != "") &
    (data["intent"] != "")
]

print("\n========== DATASET REPORT ==========")

print("\nTotal examples:", len(data))

print("\nUnique intents:", data["intent"].nunique())

print("\nIntent names:")
print(data["intent"].unique())

print("\nExamples per intent:")
print(data["intent"].value_counts())

print("\nDuplicate commands:", data.duplicated(
    subset=["text", "intent"]
).sum())

print("\nExamples per source:")
print(data["source"].value_counts())

print("\n========== REPORT COMPLETE ==========")
