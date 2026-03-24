from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Preprocess
def preprocess_input(data):
    df = pd.DataFrame([data])
    return df[["clicks", "time_spent", "email_opened"]]


# ✅ UI Home Page
@app.route("/")
def home():
    return render_template("index.html", prediction=None)


# ✅ UI Prediction
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        data = {
            "clicks": int(request.form["clicks"]),
            "time_spent": int(request.form["time_spent"]),
            "email_opened": int(request.form["email_opened"])
        }

        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return str(e)


# ✅ API (still works)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]

    return jsonify({"conversion": int(prediction)})


if __name__ == "__main__":
    app.run(debug=False)