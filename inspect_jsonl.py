
import json

files = [
    "data/raw/augmented_automation_reasoning.jsonl",
    "data/raw/intent_classification_reasoning_140.jsonl"
]

for file in files:

    print("\n================================")
    print("FILE:", file)
    print("================================")

    with open(file, "r", encoding="utf-8") as f:

        for i, line in enumerate(f):

            if i == 5:
                break

            try:
                item = json.loads(line)
                print(json.dumps(item, indent=2))

            except json.JSONDecodeError:
                print("Invalid JSON:", line)
