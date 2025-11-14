
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("IMDB_small.csv")
df = df[['review', 'sentiment']]

X = df['review']
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)


preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print("Model Accuracy:", round(acc * 100, 2), "%")


while True:
    review = input("\nEnter a review (or 'exit' to quit): ")
    if review.lower() == "exit":
        break

    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    print("Sentiment:", prediction)
