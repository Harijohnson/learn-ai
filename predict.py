import joblib

# Load trained model and vectorizer
model = joblib.load("data/spam_model.pkl")
vectorizer = joblib.load("data/vectorizer.pkl")

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