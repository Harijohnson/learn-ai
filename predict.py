import joblib

# Load trained model and vectorizer
model = joblib.load("data/spam_model.pkl")
vectorizer = joblib.load("data/vectorizer.pkl")

# New message
message = ["Meet me at the park at 5 PM today"]

# Convert message into the same numerical format
X = vectorizer.transform(message)

# Predict
prediction = model.predict(X)

# Probability
probability = model.predict_proba(X)

print("Message:", message[0])
print("Prediction:", prediction[0])
print("Probability:", probability)