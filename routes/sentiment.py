from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from well_up_ml.db.db import db  # Import db from db.db
from well_up_ml.db.models.sentiment import Sentiment
sentiment_bp = Blueprint('sentiment', __name__)

data = pd.read_csv('chat_dataset.csv')
X = data['message']
y = data['sentiment']
    # Vectorize the data
vectoriser = CountVectorizer(ngram_range=(1, 2))
X = vectoriser.fit_transform(X)
    # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


@sentiment_bp.route('/sentiment', methods=["GET"])
def sentiment():
    # Get input from body
    userInput = request.json['message']
    userInputVectorised = vectoriser.transform([userInput])
    prediction = rf_classifier.predict(userInputVectorised)

    return jsonify({"predicted_sentiment": prediction[0]})

@sentiment_bp.route('/sentiment',methods = ["GET"])
def add_new_sentiment():
   # Get input from body 
    text = request.json['text']
    sentiment = request.json['sentiment']
    new_sentiment = Sentiment(sentiment_text=text, sentiment_label=sentiment)
    db.session.add(new_sentiment)
    db.session.commit()
    return jsonify({"message": "Sentiment added successfully"},201)


@sentiment_bp.route('/sentiment/accuracy',methods=["GET"])
def sentiment_accuracy():
    #Test the model 
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return jsonify({"accuracy": accuracy, "report": report})
@sentiment_bp.route('/sentiment/setup',methods=["POST"])
def retrain():
    #read from the DB 
    sentiments = Sentiment.query.all()

    global rf_classifier
    global vectoriser
    global X_train
    global y_train
    #use data from the DB to retrain the model
    X = [sentiment.sentiment_text for sentiment in sentiments]
    y = [sentiment.sentiment_label for sentiment in sentiments]
    vectoriser = CountVectorizer(ngram_range=(1, 2))
    X = vectoriser.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return jsonify({"message": "Model retrained successfully"}, 200)

@sentiment_bp.route('/sentiment/seed',methods=["POST"])
def seed_db():
    #Drop all the data from the DB
    Sentiment.query.delete()
    db.session.commit()
    #Seed the DB with the original data
    data = pd.read_csv('chat_dataset.csv')
    message = data['message']
    sentiment = data['sentiment']
    for i in range(len(message)):
        new_sentiment = Sentiment(sentiment_text=message[i], sentiment_label=sentiment[i])
        db.session.add(new_sentiment)

    db.session.commit()

    retrain()
    return jsonify({"message": "DB seeded successfully"}, 200)

