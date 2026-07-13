# 🛡️ News Predictor — Modern NLP Classification Engine

News Predictor is an end-to-end Machine Learning web application designed to evaluate text credibility in real-time. Powered by Natural Language Processing (NLP) techniques, the platform processes incoming news headlines and text bodies through a clean, tokenized data pipeline to predict whether an article structurally aligns with authentic, informative journalism or deceptive, sensationalized reporting.

---

## 🚀 Key Features

- **Text Pipeline Processing:** Implements customized NLTK token cleaning, including lowercase conversions, special character removal (`regex`), non-informative stopword filtering, and Porter Stemming morphology.
- **Advanced Vectorization:** Transforms raw strings into numeric arrays using optimized TF-IDF (`Term Frequency-Inverse Document Frequency`) text extraction matrices.
- **Premium Web Dashboard:** Features a modern, enterprise-ready dark-mode console built entirely with **Flask** and styled dynamically with **Tailwind CSS**.
- **Intuitive Visual Feedback:** Leverages responsive UI card components that instantly display classified outputs paired with clear verification diagnostics.

---

## ⚙️ Core Pipeline Logic

The machine learning model is trained on a unified structural feature vector matching the pattern:

$$\text{Feature String} = \text{Article Title} + \text{" "} + \text{Article Body Text}$$

When a request passes through the console form, the incoming text is immediately processed and normalized by the backend pipeline through the following sequence:

1. **Text Cleansing:** Strips out all numbers, special characters, and punctuation marks using regular expressions.
2. **Case Normalization:** Converts alphabetical cases to absolute lowercase tokens to ensure uniform vocabulary matches.
3. **Stop-words Filtering:** Filters out common English language stop-words (e.g., *the, is, at, which, an*) that don't add semantic weight to credibility detection.
4. **Morphological Stemming:** Collapses words down to their base root forms using the **Porter Stemmer** algorithm (e.g., *running, runs, ran* all reduce to *run*).
5. **Feature Vectorization:** Passes the clean string into the fitted `TfidfVectorizer` to create a numerical array.
6. **Model Prediction:** Inputs the vectorized array into the core classifier matrix for a real-time predictive probability output (`0` for Real News, `1` for Fake News).

---

## 🛠️ Tech Stack & Dependencies

### Machine Learning & Pipeline:
- **Python 3.x**
- **NumPy & Pandas** (Data preparation and feature handling)
- **Scikit-Learn** (`TfidfVectorizer` & Classifier model)
- **NLTK** (Natural Language Toolkit for morphological word stemming)

### Web Framework & UI:
- **Flask** (Backend micro-framework serving routing requests)
- **Tailwind CSS** (Responsive utility-first frontend layout via CDN)
- **Jinja2** (Dynamic layout rendering engines)

---

## 📂 Project Architecture

```text
news-predictor/
│
├── app.py                      # Flask web application engine
├── fake_news_model.pkl          # Serialized (pickled) predictive ML model
├── tfidf_vectorizer.pkl        # Serialized TF-IDF text transformer matrix
├── Fake_news_detection_pipeline.ipynb  # Interactive training & validation notebook
│
├── templates/                  # Frontend UI templates
│   └── index.html              # Responsive dark-mode console layout
│
└── README.md                   # System configuration & operational documentation
