from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# Training data
# texts = [
#     "Win free money",
#     "Claim your prize",
#     "You won a free gift",
#     "Click here to win cash",
#     "Meeting at 5 PM",
#     "Can you call me?",
#     "Let's meet tomorrow",
#     "Please send me the report",
# ]

# labels = [
#     "spam",
#     "spam",
#     "spam",
#     "spam",
#     "ham",
#     "ham",
#     "ham",
#     "ham",
# ]

# texts  = [t for t, label in spam_ham.data]
# labels = [label for t, label in spam_ham.data]

# Load data from spam.csv (columns: v1 = label, v2 = text)
import pandas as pd

df = pd.read_csv(DATA_DIR / "spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].dropna()

texts  = df["v2"].tolist()
labels = df["v1"].tolist()


texts_train, texts_test, labels_train, labels_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


# Convert text into numbers
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(texts)
X_train = vectorizer.fit_transform(texts_train)

# Create model
model = LogisticRegression()

# Train model
# model.fit(X, labels)
model.fit(X_train, labels_train)


words = vectorizer.get_feature_names_out()
weights = model.coef_[0]

for word, weight in zip(words, weights):
    print(f"{word:10} {weight:.4f}")

	
# Check what the model learned
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nWeights:")
print(model.coef_)

# Save both
joblib.dump(model, DATA_DIR / "spam_model.pkl")
joblib.dump(vectorizer, DATA_DIR / "vectorizer.pkl")

print("\nModel saved successfully.")


# Predict on test data
X_test = vectorizer.transform(texts_test)

predictions = model.predict(X_test)

# Compare predictions with actual labels
accuracy = accuracy_score(labels_test, predictions)

print("Accuracy:", accuracy)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels_test, predictions)

print("Confusion Matrix:")
print(cm)


for text, actual, predicted in zip(texts_test, labels_test, predictions):
    if actual != predicted:
        print("\nMessage:", text)
        print("Actual:", actual)
        print("Predicted:", predicted)




message = ["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a 1500 Bonus Prize, call 09066364589"]

X = vectorizer.transform(message)

print(X.toarray())

feature_names = vectorizer.get_feature_names_out()
values = X.toarray()[0]

for word, value in zip(feature_names, values):
    if value > 0:
        print(word, value)

print("Intercept:", model.intercept_)
probability = model.predict_proba(X)

print("Probability:", probability)


# tests = [
#     "Win free money now",
#     "Can you call me tomorrow?"
# ]

tests = [
    "Please call me to claim your free prize",
    "Can you call me about the meeting?",
]
for message in tests:
    X = vectorizer.transform([message])

    print("\nMessage:", message)
    print("Prediction:", model.predict(X)[0])
    print("Probability:", model.predict_proba(X)[0])