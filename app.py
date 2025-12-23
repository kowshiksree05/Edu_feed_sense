from flask import Flask, render_template, request, redirect, url_for, flash
import os
from datetime import datetime
import pandas as pd
from transformers import pipeline
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import flair

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "1234"  

# Load faculty list from faculty.json
def load_faculty():
    if os.path.exists('faculty.json'):
        with open('faculty.json', 'r') as f:
            data = json.load(f)
            return data.get('faculty', [])
    return []

# Save faculty list to faculty.json
def save_faculty(faculty_list):
    with open('faculty.json', 'w') as f:
        json.dump({'faculty': faculty_list}, f, indent=4)

# Load analyzed feedback data
def load_analyzed_feedback():
    if os.path.exists('Analyzed_feedback.json'):
        with open('Analyzed_feedback.json', 'r') as f:
            return json.load(f)
    return {}

# Save analyzed feedback data
def save_analyzed_feedback(analyzed_data):
    with open('Analyzed_feedback.json', 'w') as f:
        json.dump(analyzed_data, f, indent=4)




# Initialize Sentiment Analysis models
model_names = ["cardiffnlp/twitter-roberta-base-sentiment","distilbert-base-uncased","nlptown/bert-base-multilingual-uncased-sentiment","google/electra-base-discriminator"]
hf_models = [pipeline('sentiment-analysis', model=name) for name in model_names] #hugged face models 
flair_sentiment_analyzer = flair.models.TextClassifier.load('en-sentiment')
vader_analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis functions
def flair_predict(text):
    sentence = flair.data.Sentence(text)
    flair_sentiment_analyzer.predict(sentence)
    score = sentence.labels[0].score
    label = sentence.labels[0].value
    if score >= 0.05:
        return 'POSITIVE'
    elif score <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def vader_predict(text):
    sentiment = vader_analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'POSITIVE'
    elif sentiment['compound'] <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def textblob_predict(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'POSITIVE'
    elif polarity < 0:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def update_analyzed_feedback(faculty_name, sentiment_summary):
    faculty_list = load_faculty()
    analyzed_feedback = -1  # Initialize as neutral value
    
    # Logic for determining final sentiment based on ties
    if (sentiment_summary.get('POSITIVE', 0) == sentiment_summary.get('NEGATIVE', 0) and
        sentiment_summary.get('NEGATIVE', 0) == sentiment_summary.get('NEUTRAL', 0)):
        # If all sentiments are equal, return NEUTRAL
        analyzed_feedback = 2  # Neutral
    elif sentiment_summary.get('POSITIVE', 0) == sentiment_summary.get('NEGATIVE', 0):
        # If POSITIVE and NEGATIVE are equal, return NEUTRAL
        analyzed_feedback = 2  # Neutral
    elif sentiment_summary.get('POSITIVE', 0) == sentiment_summary.get('NEUTRAL', 0):
        # If POSITIVE and NEUTRAL are equal, return POSITIVE
        analyzed_feedback = 1  # Positive
    elif sentiment_summary.get('NEGATIVE', 0) == sentiment_summary.get('NEUTRAL', 0):
        # If NEGATIVE and NEUTRAL are equal, return NEGATIVE
        analyzed_feedback = 0  # Negative
    else:
        # Default: Positive > Negative, Negative > Positive, otherwise Neutral
        if sentiment_summary.get('POSITIVE', 0) > sentiment_summary.get('NEGATIVE', 0):
            analyzed_feedback = 1  # Positive
        elif sentiment_summary.get('NEGATIVE', 0) > sentiment_summary.get('POSITIVE', 0):
            analyzed_feedback = 0  # Negative
        else:
            analyzed_feedback = 2  # Neutral if no clear positive or negative winner
    
    # Update the faculty list with the new analyzed feedback
    for faculty in faculty_list:
        if faculty['name'].lower() == faculty_name.lower():
            faculty['analyzed_feedback'] = analyzed_feedback
            break
    save_faculty(faculty_list)


def aggregate_predictions(text):
    predictions = []
    for model in hf_models:
        pred = model(text)[0]
        predictions.append(pred['label'])

    flair_label = flair_predict(text)
    predictions.append(flair_label)

    vader_label = vader_predict(text)
    predictions.append(vader_label)

    textblob_label = textblob_predict(text)
    predictions.append(textblob_label)

    label_counts = {label: predictions.count(label) for label in set(predictions)}
    
    # Ensure the final label is not "Unknown"
    final_label = max(label_counts, key=label_counts.get)

    # Adjust if there is a clear tie
    if final_label == 'LABEL_0':
        return 'NEGATIVE'
    elif final_label == 'LABEL_1':
        return 'NEUTRAL'
    elif final_label == 'LABEL_2':
        return 'POSITIVE'
    else:
        return final_label  # Ensure no "Unknown" can occur




@app.route('/')
def index():
    faculty_list = load_faculty()
    return render_template('index.html', faculty_list=faculty_list)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename.endswith('.csv'):
        faculty_name = file.filename.rsplit('.', 1)[0].lower()
        faculty_list = load_faculty()
        if faculty_name in [f['name'].lower() for f in faculty_list]:
            file.save(os.path.join("uploads", file.filename))
            flash('File uploaded successfully', 'success')
            return redirect(url_for('process_csv', filename=file.filename))
        else:
            flash('Faculty name in file does not match any existing faculty', 'danger')
            return redirect(url_for('index'))
    flash('Only CSV files are allowed', 'danger')
    return redirect(url_for('index'))

@app.route('/process_csv/<filename>')
def process_csv(filename):
    file_path = os.path.join('uploads', filename)
    df = pd.read_csv(file_path)
    df['Sentiment'] = df['feedback'].apply(lambda x: aggregate_predictions(x))
    sentiment_summary = df['Sentiment'].value_counts().to_dict()

    faculty_name = filename.rsplit('.', 1)[0].lower()
    analyzed_feedback_data = load_analyzed_feedback()
    analyzed_feedback_data[faculty_name] = sentiment_summary
    save_analyzed_feedback(analyzed_feedback_data)
    update_analyzed_feedback(faculty_name, sentiment_summary)
    system_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chart_data = {
        'labels': list(sentiment_summary.keys()),
        'data': list(sentiment_summary.values())
    }
    return render_template('result.html', df=df, sentiment_summary=sentiment_summary, chart_data=chart_data,faculty_name=faculty_name,system_date=system_date)

@app.route('/add', methods=['POST'])
def add_faculty():
    name = request.form.get('name')
    faculty_list = load_faculty()
    if name and name.lower() not in [f['name'].lower() for f in faculty_list]:
        faculty_list.append({'name': name, 'analyzed_feedback': -1})
        save_faculty(faculty_list)
        flash(f"Faculty '{name}' added successfully!", 'success')
    else:
        flash(f"Faculty '{name}' already exists or name is invalid", 'danger')
    return redirect(url_for('index'))

@app.route('/delete', methods=['POST'])
def delete_faculty():
    name = request.form.get('faculty_name')
    faculty_list = load_faculty()
    faculty_to_remove = next((f for f in faculty_list if f['name'] == name), None)
    if faculty_to_remove:
        faculty_list.remove(faculty_to_remove)
        save_faculty(faculty_list)
        flash(f"Faculty '{name}' deleted successfully!", 'success')
    else:
        flash(f"Faculty '{name}' not found", 'danger')
    return redirect(url_for('index'))

@app.route('/edit', methods=['POST'])
def edit_faculty():
    old_name = request.form.get('faculty_name')
    new_name = request.form.get('new_name')
    faculty_list = load_faculty()
    
    for faculty in faculty_list:
        if faculty['name'].lower() == old_name.lower():
            if new_name and not any(f['name'].lower() == new_name.lower() for f in faculty_list):
                faculty['name'] = new_name
                save_faculty(faculty_list)
                flash(f"Faculty '{old_name}' updated to '{new_name}'!", 'success')
                return redirect(url_for('index'))
            else:
                flash(f"New name is either invalid or already exists", 'danger')
                return redirect(url_for('index'))
    
    flash(f"Faculty '{old_name}' not found", 'danger')
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)

