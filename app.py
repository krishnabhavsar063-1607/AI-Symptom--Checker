from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

# Split data
X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        fever = int(request.form.get("fever", 0))
        cough = int(request.form.get("cough", 0))
        headache = int(request.form.get("headache", 0))
        fatigue = int(request.form.get("fatigue", 0))
        body_pain = int(request.form.get("body_pain", 0))

        input_data = [[fever, cough, headache, fatigue, body_pain]]
        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)