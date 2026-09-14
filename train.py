from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import spam_ham
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

texts  = [t for t, label in spam_ham.data]
labels = [label for t, label in spam_ham.data]
# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Create model
model = LogisticRegression()

# Train model
model.fit(X, labels)


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
joblib.dump(model, "data/spam_model.pkl")
joblib.dump(vectorizer, "data/vectorizer.pkl")

print("\nModel saved successfully.")