from datasets import load_dataset
from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:5000]"
)

with open(DATA_DIR / "tinystories.txt", "w", encoding="utf-8") as file:
    for item in dataset:
        file.write(item["text"])
        file.write("\n")

print("Stories:", len(dataset))
print("Saved to: data/tinystories.txt")