# 🛡️ News Predictor — Modern NLP Classification Engine

News Predictor is an end-to-end Machine Learning web application designed to evaluate text credibility in real-time. Powered by Natural Language Processing (NLP) techniques, the platform processes incoming news headlines and text bodies through a clean, tokenized data pipeline to predict whether an article structurally aligns with authentic, informative journalism or deceptive, sensationalized reporting.

---

## 🚀 Key Features

- **Text Pipeline Processing:** Implements customized NLTK token cleaning, including lowercase conversions, special character removal (`regex`), non-informative stopword filtering, and Porter Stemming morphology.
- **Advanced Vectorization:** Transforms raw strings into numeric arrays using optimized TF-IDF (`Term Frequency-Inverse Document Frequency`) text extraction matrices.
- **Premium Web Dashboard:** Features a modern, enterprise-ready dark-mode console built entirely with **Flask** and styled dynamically with **Tailwind CSS**.
- **Intuitive Visual Feedback:** Leverages responsive UI card components that instantly display classified outputs paired with clear verification diagnostics.

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
