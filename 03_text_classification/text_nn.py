import csv

import joblib
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# =========================================================
# 1. Load SMS dataset
# =========================================================

# texts = []
# labels = []

# with open("data/SMSSpamCollection", "r", encoding="utf-8") as file:
#     reader = csv.reader(file, delimiter="\t")

#     for row in reader:
#         if len(row) != 2:
#             continue

#         label, message = row

#         texts.append(message)

#         if label == "ham":
#             labels.append(0)
#         elif label == "spam":
#             labels.append(1)

# Load data from spam.csv (columns: v1 = label, v2 = text)
import pandas as pd
texts = []
labels = []
df = pd.read_csv(DATA_DIR / "spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].dropna()

for label, message in zip(df["v1"], df["v2"]):

    if label == "ham":
        labels.append(0)
        texts.append(message)
    elif label == "spam":
        labels.append(1)
        texts.append(message)


print("Total messages:", len(texts))


# =========================================================
# 2. Train / test split
# =========================================================

texts_train, texts_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training messages:", len(texts_train))
print("Test messages:", len(texts_test))


# =========================================================
# 3. Text -> TF-IDF numbers
# =========================================================

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

print("Number of features:", X_train.shape[1])


# =========================================================
# 4. Convert to PyTorch tensors
# =========================================================

X_train = torch.tensor(
    X_train.toarray(),
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test.toarray(),
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# =========================================================
# 5. Create mini-batches
# =========================================================

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# =========================================================
# 6. Neural network
# =========================================================

input_size = X_train.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)


# =========================================================
# 7. Loss + optimizer
# =========================================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================================================
# 8. Training
# =========================================================

epochs = 15

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        # Forward pass
        output = model(batch_X)

        # Calculate loss
        loss = loss_function(output, batch_y)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.4f}"
    )


# =========================================================
# 9. Evaluate on unseen test data
# =========================================================

model.eval()

with torch.no_grad():

    output = model(X_test)

    predictions = torch.argmax(output, dim=1)


# =========================================================
# 10. Metrics
# =========================================================

accuracy = accuracy_score(
    y_test.numpy(),
    predictions.numpy()
)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test.numpy(),
        predictions.numpy()
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test.numpy(),
        predictions.numpy(),
        target_names=["ham", "spam"]
    )
)


# =========================================================
# 11. Save model
# =========================================================

joblib.dump(
    vectorizer,
    DATA_DIR / "tfidf_vectorizer.pkl"
)

torch.save(
    model.state_dict(),
    DATA_DIR / "spam_nn_model.pth"
)

print("\nModel saved successfully.")


# =========================================================
# 12. Test a new message
# =========================================================

test_message = [
    "Congratulations! You won a free prize. Call now!"
]

test_vector = vectorizer.transform(test_message)

test_tensor = torch.tensor(
    test_vector.toarray(),
    dtype=torch.float32
)

with torch.no_grad():

    test_output = model(test_tensor)

    test_prediction = torch.argmax(
        test_output,
        dim=1
    )

    test_probability = torch.softmax(
        test_output,
        dim=1
    )


print("\nNew message:")
print(test_message[0])

print("Prediction:", end=" ")

if test_prediction.item() == 0:
    print("ham")
else:
    print("spam")

print("Probability:")
print(
    "ham :",
    test_probability[0][0].item()
)

print(
    "spam:",
    test_probability[0][1].item()
)