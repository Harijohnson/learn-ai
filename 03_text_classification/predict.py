import joblib
from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Load trained model and vectorizer
model = joblib.load(DATA_DIR / "spam_model.pkl")
vectorizer = joblib.load(DATA_DIR / "vectorizer.pkl")

# New message
message = ["WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]

# Convert message into the same numerical format
X = vectorizer.transform(message)

# Predict
prediction = model.predict(X)

# Probability
probability = model.predict_proba(X)

print("Message:", message[0])
print("Prediction:", prediction[0])
print("Probability:", probability)

from sklearn.metrics import accuracy_score

