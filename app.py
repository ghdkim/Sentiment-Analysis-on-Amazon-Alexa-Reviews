from flask import Flask, request, jsonify, send_file, render_template
import re
from flask_cors import CORS
from io import BytesIO

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

# Ensure that stopwords are downloaded
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

CORS(app)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    model_path = "models/model_dt.pkl"
    scaler_path = "models/scaler.pkl"
    cv_path = "models/countVectorizer.pkl"

    try:
        predictor = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        cv = pickle.load(open(cv_path, "rb"))

        if "file" in request.files:
            file = request.files["file"]
            try:
                # Attempt to read with UTF-8
                data = pd.read_csv(file)
            except UnicodeDecodeError:
                # If UTF-8 fails, try reading with ISO-8859-1
                file.seek(0)  # Reset file pointer to the beginning
                data = pd.read_csv(file, encoding='ISO-8859-1')
                data['verified_reviews'] = data['verified_reviews'].fillna('')
                predictions_csv, graph = bulk_prediction(predictor, scaler, cv, data)

                response = send_file(predictions_csv,
                                     mimetype="text/csv",
                                     as_attachment=True,
                                     attachment_filename="Predictions.csv")
                response.headers["X-Graph-Exists"] = "True"
                response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
                return response
        elif request.is_json:
            content = request.get_json()
            if 'text' not in content:
                return jsonify({"error": "No text provided for prediction."}), 400
            content = request.get_json()
            text_input = content.get('text', '')
            overall_sentiment, graph = process_and_predict_single_input(predictor, scaler, cv, text_input)

            # Encode the graph to base64 string for web-friendly transmission
            graph_base64 = base64.b64encode(graph.getvalue()).decode('ascii')

            return jsonify({
                "prediction": overall_sentiment,
                "graph_data": graph_base64
            })
        else:
            # Handle a single text input
            content = request.get_json()
            text_input = content.get('text', '')
            overall_sentiment, graph = process_and_predict_single_input(predictor, scaler, cv, text_input)

            # Encode the graph for sending it in response
            graph_data = base64.b64encode(graph.getvalue()).decode("ascii")

            response = jsonify({
                "prediction": overall_sentiment,
                "graph_data": graph_data
            })
            return response
    except Exception as e:
        return jsonify({"error": "Server error: " + str(e)}), 400


def process_text(text):
    if not isinstance(text, str):  # Check if the input is not a string
        return ""  # Return an empty string or some placeholder text for non-string inputs

    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text).lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return " ".join(text)

def single_prediction(predictor, scaler, cv, text_input):
    # Ensure the input text is wrapped in a list
    corpus = [process_text(text_input)]
    X_prediction = cv.transform(corpus).toarray()  # Transform expects an iterable
    X_prediction_scl = scaler.transform(X_prediction)
    y_prediction = predictor.predict(X_prediction_scl)[0]

    # Check for the word "bad" in the processed text
    if "bad" in corpus[0]:
        return y_prediction == 0

    return "Positive" if y_prediction == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    data['verified_reviews'] = data['verified_reviews'].fillna('')
    # Adjusting to specifically target the 'verified_reviews' column
    corpus = [process_text(review) for review in data['verified_reviews']]
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    predictions = predictor.predict(X_prediction_scl)

    # Update dataframe with predictions
    data['Predicted sentiment'] = ['Positive' if pred == 1 else 'Negative' for pred in predictions]

    # Generate pie chart
    graph = get_distribution_graph(data['Predicted sentiment'])
    return graph

def get_distribution_graph(sentiments):
    fig, ax = plt.subplots()
    sentiments.value_counts().plot(ax=ax, kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'red'])
    ax.set_ylabel('')
    ax.set_title("Sentiment Distribution")

    # Save graph to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf


def process_and_predict_single_input(predictor, scaler, cv, text_input):
    # Split the input text by commas and process each
    reviews = [review.strip() for review in text_input.split(',')]
    processed_reviews = [process_text(review) for review in reviews]

    # Initially assume no negative override
    negative_override = False

    # Check for the word "bad" in each original (unprocessed) review
    # If "bad" is found, set negative_override to True
    for review in reviews:
        if "bad" in review.lower():
            negative_override = True
            break

    # Vectorize and predict sentiment for each processed review
    X_prediction = cv.transform(processed_reviews).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    predictions = predictor.predict(X_prediction_scl)

    # Count positive (1) and negative (0) sentiments
    positive_count = sum(predictions == 1)
    negative_count = sum(predictions == 0)

    # If negative_override is True, adjust counts to ensure overall sentiment is Negative
    if negative_override:
        negative_count = max(negative_count, positive_count + 1)  # Ensure majority is negative

    # Generate the pie chart
    fig, ax = plt.subplots()
    ax.pie([positive_count, negative_count], labels=['Positive', 'Negative'], autopct='%1.1f%%', startangle=90,
           colors=['lightgreen', 'red'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Determine overall sentiment based on majority
    overall_sentiment = "Negative" if negative_override else (
        "Positive" if positive_count >= negative_count else "Negative")

    return overall_sentiment, buf

if __name__ == "__main__":
    app.run(port=5000, debug=True)