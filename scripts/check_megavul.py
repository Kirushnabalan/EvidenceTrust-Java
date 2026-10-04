import json
from pathlib import Path


JSON_PATH = Path("dataset/raw/megavul/megavul.json")


with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


print("JSON file:", JSON_PATH)
print("Top-level type:", type(data).__name__)


if isinstance(data, list):
    print("Number of records:", len(data))

    if len(data) > 0:
        print("\nFirst record:")
        print(data[0])

        if isinstance(data[0], dict):
            print("\nFirst record keys:")
            for key in data[0].keys():
                print("  -", key)

elif isinstance(data, dict):
    print("\nTop-level keys:")
    for key in data.keys():
        print("  -", key)
