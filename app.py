from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Configure Flask to search for HTML templates in the immediate root workspace directory
app = Flask(__name__, template_folder='.')

# Preload text processing tools
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

port_stem = PorterStemmer()

# Load model and vectorizer once on startup
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    input_data = {"title": "", "text": ""}

    if request.method == 'POST':
        # Retrieve form data
        title = request.form.get('title', '')
        text = request.form.get('text', '')
        
        # Preserve input data to fill the form back up
        input_data = {"title": title, "text": text}

        if title.strip() and text.strip():
            # Remapped strictly to title + space + text
            combined_content = f"{title} {text}" 
            clean_text = preprocess_text(combined_content)
            
            # Vectorize and Predict
            vectorized_input = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized_input)[0]
            
            # Formulate the response object
            if prediction == 1:
                prediction_result = {
                    "status": "fake",
                    "title": "🚨 Verification Failure: Highly Probable Fake News",
                    "desc": "Our machine learning pipeline has identified specific linguistic patterns, structural biases, or contextual vocabulary heavily aligned with unverified or sensationalized reporting."
                }
            else:
                prediction_result = {
                    "status": "real",
                    "title": "✅ Verification Success: Highly Probable Real News",
                    "desc": "This text structurally matches baseline informative patterns and linguistic density indexes common in standard credible journalism."
                }

    return render_template('index.html', result=prediction_result, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
