
import csv
import json
from pathlib import Path

DATA_FOLDER = Path("data/raw")

print("\n========== JARVIS DATASET INSPECTION ==========\n")

for file_path in DATA_FOLDER.iterdir():

    if not file_path.is_file():
        continue

    print("\nFILE:", file_path.name)

    try:

        if file_path.suffix == ".csv":

            with open(file_path, "r", encoding="utf-8-sig", newline="") as file:

                reader = csv.DictReader(file)

                print("Columns:", reader.fieldnames)

                for i, row in enumerate(reader):
                    if i == 3:
                        break
                    print("Sample:", row)

        elif file_path.suffix == ".jsonl":

            with open(file_path, "r", encoding="utf-8") as file:

                for i, line in enumerate(file):

                    if i == 3:
                        break

                    print("Sample:", json.loads(line))

    except Exception as error:

        print("Error:", error)

print("\n========== INSPECTION COMPLETE ==========")
