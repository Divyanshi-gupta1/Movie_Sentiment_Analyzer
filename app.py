from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

df = pd.read_csv("IMDB_small.csv")
df = df[['review', 'sentiment']]

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        review_text = request.form.get("review")
        if review_text:
            review_vec = vectorizer.transform([review_text])
            prediction = model.predict(review_vec)[0]
            sentiment = prediction
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)