
import pandas as pd
import json
from pathlib import Path

folder = Path("data/raw")

all_data = []

# 1. Read CSV files
for file in folder.glob("*.csv"):

    df = pd.read_csv(file)

    df = df[["text", "intent"]]

    all_data.append(df)


# 2. Read JSONL files
for file in folder.glob("*.jsonl"):

    rows = []

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            rows.append({
                "text": item.get("instruction", ""),
                "intent": item.get("intent", "")
            })

    all_data.append(pd.DataFrame(rows))


# 3. Combine datasets
data = pd.concat(all_data, ignore_index=True)

# 4. Clean text
data["text"] = data["text"].fillna("").str.strip()
data["intent"] = data["intent"].fillna("").str.strip()

data = data[
    (data["text"] != "") &
    (data["intent"] != "")
]

# 5. Find conflicting commands
conflicts = data.groupby("text")["intent"].nunique()

conflict_texts = conflicts[conflicts > 1].index

print("\nConflicting commands:")

for command in conflict_texts:

    print("\nCommand:", command)

    print(
        data[data["text"] == command][["text", "intent"]]
        .drop_duplicates()
        .to_string(index=False)
    )

# 6. Remove conflicting commands
data = data[~data["text"].isin(conflict_texts)]

# 7. Remove duplicate examples
data = data.drop_duplicates(
    subset=["text", "intent"]
)

# 8. Save clean dataset
Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

data.to_csv(
    "data/processed/jarvis_smart_home.csv",
    index=False
)

# 9. Final report
print("\n========== CLEAN DATASET ==========")

print("Clean examples:", len(data))

print("Unique intents:", data["intent"].nunique())

print("\nExamples per intent:")
print(data["intent"].value_counts())

print("\nSaved successfully!")
print("data/processed/jarvis_smart_home.csv")
