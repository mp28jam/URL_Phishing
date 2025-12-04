# Flask web framework
from flask import Flask, render_template, request
# To load models
import joblib
# Import prediction function we created
from URL_Detector import URLpredict

# Constructor for Flask
app = Flask(__name__)

model = joblib.load("trained_model.joblib")        # Load trained classifier
vectorizer = joblib.load("vectorizer.joblib")      # Load TF-IDF vectorizer

# ===============================================================
# Handle GET Request
# Return index.html template when users open up the program
# ===============================================================

@app.route("/")
def home():
    return render_template("index.html", accuracy=None, training_time=None, class_report=None, prediction=None, url="None")

# ===============================================================
# Handle POST Request
# Return index.html template when users open up the program
# ===============================================================
@app.route("/predict", methods=["POST"])
def predict():
    # The URL that users type in
    url = request.form["url_input"]
    # Predict if URL is malicious using trained model
    accuracy, training_time, class_report, prediction = URLpredict(url)
    # Load index.html with the result
    return render_template("index.html", accuracy=accuracy, training_time=training_time, class_report=class_report, prediction=prediction, url=url)

# Run the program
if __name__=='__main__':
   app.run()