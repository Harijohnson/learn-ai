from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this",
    "This is amazing",
    "I hate this",
    "This is terrible",
]

labels = [
    "positive",
    "positive",
    "negative",
    "negative",
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(X.toarray())

model = LogisticRegression()


model.fit(X, labels)

print(model.coef_)

test = ["I love this product"]

X_test = vectorizer.transform(test)

prediction = model.predict(X_test)

print(prediction)