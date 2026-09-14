from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# clasification of text into positive and negative sentiments
# texts = [
#     "I love this",
#     "This is amazing",
#     "I hate this",
#     "This is terrible",
#     "Hari is great",
#     "Hari is good",
#     "Hari is bad",
# ]

# labels = [
#     "positive",
#     "positive",
#     "negative",
#     "negative",
#     "positive",
#     "positive",
#     "negative",
# ]


# ham or spam classification

texts = [
	"Win free money",
	"Claim your prize",
	"Meeting at 5 PM",
	"Can you call me?"
]
labels = [
	"spam",
	"spam",
	"ham",
	"ham"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(X.toarray())

model = LogisticRegression()


model.fit(X, labels)

print(model.coef_)

test = ["Meet me at park at 5PM today"]

X_test = vectorizer.transform(test)

prediction = model.predict(X_test)

print(prediction)