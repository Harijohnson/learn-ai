from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train"
)

print(dataset)
print(dataset[0])