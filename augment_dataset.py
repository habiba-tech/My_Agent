
import pandas as pd
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

input_file = "data/processed/jarvis_smart_home.csv"
output_file = "data/processed/jarvis_smart_home_augmented.csv"

TARGET_PER_INTENT = 100

# ==========================================
# 2. LOAD AND CLEAN DATASET
# ==========================================

data = pd.read_csv(input_file)

data["text"] = data["text"].fillna("").astype(str).str.strip()
data["intent"] = data["intent"].fillna("").astype(str).str.strip()

data = data[
    (data["text"] != "") &
    (data["intent"] != "")
].copy()

data["_key"] = data["text"].str.lower().str.strip()

# Remove commands that have conflicting labels
conflicts = data.groupby("_key")["intent"].nunique()
conflict_keys = set(conflicts[conflicts > 1].index)

data = data[~data["_key"].isin(conflict_keys)]
data = data.drop_duplicates(subset="_key")

# ==========================================
# 3. DEVICE NAMES
# ==========================================

devices = {
    "light_on": [
        "lights", "room lights", "bedroom lights",
        "living room lights", "lamp", "bedroom lamp",
        "hall lights", "main lights", "all lights",
        "ceiling lights"
    ],

    "light_off": [
        "lights", "room lights", "bedroom lights",
        "living room lights", "lamp", "bedroom lamp",
        "hall lights", "main lights", "all lights",
        "ceiling lights"
    ],

    "ac_on": [
        "AC", "air conditioner", "air conditioning",
        "cooling system", "bedroom AC", "living room AC",
        "AC unit", "cooling unit", "room air conditioner",
        "central AC"
    ],

    "ac_off": [
        "AC", "air conditioner", "air conditioning",
        "cooling system", "bedroom AC", "living room AC",
        "AC unit", "cooling unit", "room air conditioner",
        "central AC"
    ],

    "vent_on": [
        "ventilation system", "vent", "room ventilation",
        "air ventilator", "kitchen exhaust", "bathroom exhaust",
        "exhaust system", "air circulation system",
        "air vent", "ventilation"
    ],

    "vent_off": [
        "ventilation system", "vent", "room ventilation",
        "air ventilator", "kitchen exhaust", "bathroom exhaust",
        "exhaust system", "air circulation system",
        "air vent", "ventilation"
    ],

    "fan_on": [
        "fan", "ceiling fan", "table fan", "bedroom fan",
        "living room fan", "office fan", "pedestal fan",
        "wall fan", "room fan", "portable fan"
    ],

    "fan_off": [
        "fan", "ceiling fan", "table fan", "bedroom fan",
        "living room fan", "office fan", "pedestal fan",
        "wall fan", "room fan", "portable fan"
    ],

    "heater_on": [
        "heater", "room heater", "electric heater",
        "space heater", "water heater", "heating system",
        "bedroom heater", "portable heater",
        "central heating", "heat unit"
    ],

    "heater_off": [
        "heater", "room heater", "electric heater",
        "space heater", "water heater", "heating system",
        "bedroom heater", "portable heater",
        "central heating", "heat unit"
    ],

    "multiplug_on": [
        "multiplug", "power strip", "extension board",
        "socket strip", "power board", "plug board",
        "multi-plug", "smart plug", "power outlet strip",
        "extension socket"
    ],

    "multiplug_off": [
        "multiplug", "power strip", "extension board",
        "socket strip", "power board", "plug board",
        "multi-plug", "smart plug", "power outlet strip",
        "extension socket"
    ],

    "tv_on": [
        "TV", "television", "smart TV", "living room TV",
        "bedroom TV", "television screen", "TV set",
        "smart television", "main TV", "family room TV"
    ],

    "tv_off": [
        "TV", "television", "smart TV", "living room TV",
        "bedroom TV", "television screen", "TV set",
        "smart television", "main TV", "family room TV"
    ]
}

# ==========================================
# 4. DIRECT COMMAND TEMPLATES
# ==========================================

templates = [
    "{action} the {device}",
    "Please {action} the {device}",
    "Can you {action} the {device}",
    "Could you {action} the {device} for me",
    "I want to {action} the {device}",
    "I need you to {action} the {device}",
    "Kindly {action} the {device}",
    "Would you {action} the {device}",
    "Make sure to {action} the {device}",
    "Can you please {action} the {device}",
    "Please make the {device} {state}",
    "I would like the {device} {state}",
    "Set the {device} to {state}"
]

actions = {
    "on": [
        "turn on", "switch on", "activate",
        "power on", "start", "enable"
    ],

    "off": [
        "turn off", "switch off", "deactivate",
        "power off", "shut off", "disable"
    ]
}

# ==========================================
# 5. INDIRECT NATURAL LANGUAGE EXAMPLES
# ==========================================

natural_sentences = {

    "light_on": [
        "It's too dark in here",
        "I can barely see anything",
        "This room needs more light",
        "The room is very dark",
        "I need some light in this room",
        "It's difficult to see in here",
        "The room could use some lighting",
        "I want the room to be brighter"
    ],

    "light_off": [
        "The room is too bright",
        "I want some darkness in here",
        "The lights are disturbing me",
        "It's bright enough in this room",
        "I don't need the lighting anymore",
        "The room should be darker",
        "I want to save electricity by switching off the lights",
        "The lights are bothering me"
    ],

    "ac_on": [
        "It's too hot in this room",
        "This room needs cooling",
        "I'm sweating because of the heat",
        "The room feels very warm",
        "I need some cool air",
        "It's getting uncomfortable because of the heat",
        "The temperature is too high",
        "I wish this room were cooler"
    ],

    "ac_off": [
        "The AC is making the room too cold",
        "I don't need cooling anymore",
        "The room is already cool enough",
        "I'm feeling cold because of the AC",
        "The cooling is too strong",
        "This room is getting too chilly from the AC",
        "The AC is making me uncomfortable",
        "I want to stop the cooling"
    ],

    "vent_on": [
        "The air in this room feels stale",
        "This room needs better ventilation",
        "The kitchen is full of smoke",
        "There is too much cooking smell",
        "The bathroom needs air circulation",
        "The room feels stuffy",
        "I need fresh air circulation",
        "The air inside feels trapped"
    ],

    "vent_off": [
        "Stop the air ventilation",
        "I don't need ventilation anymore",
        "The exhaust is no longer needed",
        "The kitchen smoke has cleared",
        "There is no need for air circulation now",
        "I want to stop the exhaust",
        "The ventilation can be stopped",
        "Turn off the air circulation"
    ],

    "fan_on": [
        "I need a breeze",
        "There is no air moving in this room",
        "I need some air on my face",
        "The room feels warm and still",
        "I wish there were more airflow from the fan",
        "I need the fan running",
        "It's uncomfortable without a breeze",
        "I could use some air movement"
    ],

    "fan_off": [
        "There is too much breeze",
        "The fan is blowing too much air",
        "I don't need the breeze anymore",
        "The moving air is making me uncomfortable",
        "The fan is disturbing me",
        "I want the airflow to stop",
        "It's too windy from the fan",
        "I no longer need the fan"
    ],

    "heater_on": [
        "I'm freezing and need some warmth",
        "This room is extremely cold and needs heating",
        "I need more warmth in this room",
        "The room feels freezing",
        "I can't get warm in here",
        "It's very chilly and I need heating",
        "I wish this room were warmer",
        "The temperature is too low and I need heat"
    ],

    "heater_off": [
        "The heater is making the room too warm",
        "I don't need heating anymore",
        "The room is warm enough now",
        "It's getting too hot from the heater",
        "I want to stop the heating",
        "The heater is no longer necessary",
        "There is enough warmth in this room",
        "The heating is making me uncomfortable"
    ],

    "multiplug_on": [
        "I need power from the extension board",
        "Please restore power to the power strip",
        "I need the connected sockets working",
        "The extension board should supply electricity",
        "I want to use the smart plug",
        "Enable electricity to the power board",
        "I need the power strip active",
        "Restore the power supply to the multiplug"
    ],

    "multiplug_off": [
        "Cut power to the extension board",
        "I don't need electricity from the power strip",
        "Stop supplying power to the multiplug",
        "Disconnect the power board",
        "The extension board is no longer needed",
        "I want to stop the power supply to the sockets",
        "Disable the smart plug",
        "Stop the electricity supply to the power strip"
    ],

    "tv_on": [
        "I want to watch something",
        "Let's watch television",
        "I want some entertainment on the TV",
        "It's time to watch my favourite show",
        "I want to catch the news",
        "I feel like watching a movie",
        "Let's put something on television",
        "I want to watch my favourite programme"
    ],

    "tv_off": [
        "I'm finished watching television",
        "I've finished my show",
        "I don't want to watch anything anymore",
        "The programme is over",
        "I'm done watching TV",
        "I want the screen to go dark",
        "There is nothing left to watch",
        "I'm going to stop watching television"
    ]
}

# ==========================================
# 6. GENERATE NEW EXAMPLES
# ==========================================

generated_rows = []

for intent, device_list in devices.items():

    state = "on" if intent.endswith("_on") else "off"

    # Add indirect natural sentences first
    for sentence in natural_sentences[intent]:

        generated_rows.append({
            "text": sentence,
            "intent": intent
        })

    # Generate direct command variations
    for device in device_list:

        for action in actions[state]:

            for template in templates:

                sentence = template.format(
                    action=action,
                    device=device,
                    state=state
                )

                generated_rows.append({
                    "text": sentence,
                    "intent": intent
                })

# ==========================================
# 7. COMBINE AND REMOVE DUPLICATES
# ==========================================

generated = pd.DataFrame(generated_rows)

combined = pd.concat(
    [
        data[["text", "intent"]],
        generated
    ],
    ignore_index=True
)

combined["text"] = combined["text"].str.strip()
combined["intent"] = combined["intent"].str.strip()

combined["_key"] = combined["text"].str.lower().str.strip()

# Remove duplicate text with the same intent
combined = combined.drop_duplicates(
    subset=["_key", "intent"]
)

# Remove any sentence that has multiple intent labels
conflicts = combined.groupby("_key")["intent"].nunique()
conflict_keys = set(conflicts[conflicts > 1].index)

combined = combined[
    ~combined["_key"].isin(conflict_keys)
]

# ==========================================
# 8. BALANCE DATASET
# ==========================================

final_parts = []

for intent in devices.keys():

    intent_data = combined[
        combined["intent"] == intent
    ].copy()

    # Keep up to 100 examples for this intent
    intent_data = intent_data.head(TARGET_PER_INTENT)

    if len(intent_data) < TARGET_PER_INTENT:
        raise ValueError(
            f"{intent} has only {len(intent_data)} examples. "
            f"Expected {TARGET_PER_INTENT}."
        )

    final_parts.append(intent_data)

final_data = pd.concat(
    final_parts,
    ignore_index=True
)

# Remove helper column
final_data = final_data[["text", "intent"]]

# ==========================================
# 9. FINAL VALIDATION
# ==========================================

final_data["_key"] = (
    final_data["text"].str.lower().str.strip()
)

if final_data["_key"].duplicated().any():
    raise ValueError("Duplicate commands found!")

if final_data["intent"].nunique() != 14:
    raise ValueError("Expected exactly 14 intents!")

if not (
    final_data["intent"].value_counts() == TARGET_PER_INTENT
).all():
    raise ValueError("Dataset is not balanced!")

# ==========================================
# 10. SAVE DATASET
# ==========================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

final_data[["text", "intent"]].to_csv(
    output_file,
    index=False
)

print("\n========== DATASET AUGMENTATION ==========")

print("Total examples:", len(final_data))
print("Unique intents:", final_data["intent"].nunique())

print("\nExamples per intent:")
print(final_data["intent"].value_counts().sort_index())

print("\nSample natural language examples:")

for intent in devices.keys():
    sample = final_data[
        final_data["intent"] == intent
    ].head(2)

    for _, row in sample.iterrows():
        print(f"{row['text']}  -->  {row['intent']}")

print("\nSaved successfully!")
print(output_file)
