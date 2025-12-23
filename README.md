# Edu Feed Sense – Educational Feedback Sentiment Analyzer

Edu Feed Sense is an AI-powered system designed to analyze student feedback on faculty using Natural Language Processing (NLP) and Machine Learning (ML) techniques.  
It extracts sentiments from feedback, builds consensus across multiple models, and generates actionable insights to help improve faculty performance and decision-making.

---

## 📁 Project Structure

```
project/
│
├── app.py                     # Main Flask application file
│
├── model.py                   # Sentiment analysis models
│
├── templates/                 # HTML templates
│   ├── index.html             # Input page
│   └── result.html            # Results display page
│
├── uploads/                   # Uploaded CSV feedback files
│
├── Analyzed_feedback.json     # Processed feedback results
│
├── faculty.json               # Faculty list and metadata
```

---

## 🚀 Features

- Collects and processes student feedback data
- Performs sentiment analysis (Positive, Negative, Neutral)
- Uses multiple ML and NLP models for robust analysis
- Builds consensus across model predictions
- Generates actionable insights for faculty performance improvement
- Supports data-driven decision-making through reports and visualizations

---

## 🛠️ Tech Stack

### Languages
- Python

### Libraries / Frameworks
- Flask
- scikit-learn
- pandas
- numpy
- matplotlib
- transformers
- nltk
- torch

### Models Used
- Multinomial Naive Bayes
- BERT and BERT-family models
- GPT-2
- RoBERTa
- XLNet
- ALBERT
- T5
- ERNIE
- XLM-R

---

## 📂 Project Workflow

1. **Data Collection**  
   Gather student feedback related to faculty performance.

2. **Preprocessing**  
   Clean and prepare text data:
   - Tokenization
   - Stopword removal
   - Lemmatization

3. **Model Training**  
   Train and fine-tune multiple ML and NLP models.

4. **Sentiment Analysis**  
   Predict sentiment for each feedback entry.

5. **Consensus Building**  
   Combine predictions from multiple models to generate overall insights.

6. **Visualization**  
   Present results using charts and graphs for easy interpretation.

---

## 📊 Example Output

- Sentiment distribution (Positive %, Negative %, Neutral %)
- Word clouds representing student opinions
- Comparative performance analysis of faculty members

---

## ▶️ Running the Application

Install required dependencies:

```
pip install flask pandas numpy scikit-learn matplotlib nltk torch transformers
```

Run the application:

```
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 📌 Notes

- Intended for academic projects, research, and learning purposes
- Uses multiple advanced NLP models, so GPU acceleration is recommended for large datasets
- Flask development server is used; not intended for production deployment

---

## 📄 License

This project is intended for educational and research purposes.
