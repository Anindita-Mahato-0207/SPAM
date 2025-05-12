from flask import Flask, render_template, request 
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/contact")  
def contact():
    return render_template("contact.html")


@app.route("/detector")
def detector():
    return render_template("detector.html")

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

spam_detector = joblib.load("spam_detector.pkl")


vectorizer = spam_detector['vectorizer']
nb_model = spam_detector['nb_model']
lr_model = spam_detector['lr_model']
rf_model = spam_detector['rf_model']
clean_text = spam_detector['clean_text']

def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    nb_prediction = nb_model.predict(vectorized_message)
    lr_prediction = lr_model.predict(vectorized_message)
    rf_prediction = rf_model.predict(vectorized_message)
    
    nb_probability = nb_model.predict_proba(vectorized_message)[0][1]
    lr_probability = lr_model.predict_proba(vectorized_message)[0][1]
    rf_probability = rf_model.predict_proba(vectorized_message)[0][1]

    return {
        "Naive Bayes": ("Spam" if nb_prediction[0] == 1 else "Not Spam", nb_probability),
        "Logistic Regression": ("Spam" if lr_prediction[0] == 1 else "Not Spam", lr_probability),
        "Random Forest": ("Spam" if rf_prediction[0] == 1 else "Not Spam", rf_probability)
    }

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]  # Message from the form
    predictions = predict_spam(message)

    # Calculate the final model and prediction here
    final_model, final_prediction = max(predictions.items(), key=lambda item: item[1][1])

    # Pass final_model and final_prediction to template too
    return render_template(
        "result.html",
        predictions=predictions,
        message=message,
        final_model=final_model,
        final_prediction=final_prediction
    )
if __name__ == "__main__":

    app.run(debug=True)


