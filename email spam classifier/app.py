# app.py
from flask import Flask, render_template, request
import pickle, re, string, os

app = Flask(__name__)

# check files exist
if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
    raise FileNotFoundError("Run train_model.py first to create spam_model.pkl and vectorizer.pkl")

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message', '')
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = None
    try:
        prob = float(model.predict_proba(vector)[0].max())
    except Exception:
        prob = None
    result = "Spam" if pred == 1 else "Not Spam"
    return render_template('result.html', prediction=result, probability=prob, user_text=message)

if __name__ == "__main__":
    print("✅ Starting Flask app...")
    app.run(debug=True)
