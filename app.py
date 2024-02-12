from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the data
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


@app.route('/sentiment', methods=["GET"])
def sentiment():
    # Get input from body
    userInput = request.json['message']
    userInputVectorised = vectoriser.transform([userInput])
    prediction = rf_classifier.predict(userInputVectorised)

    return jsonify({"predicted_sentiment": prediction[0]})

@app.route('/sentiment',methods = ["POST"])
def sentimentRetrain():
   '''Retrain the model with new data'''

@app.route('/accuracy',methods=["GET"])
def sentiment():
    #Test the model 
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return jsonify({"accuracy": accuracy, "report": report})

if __name__ == '__main__':
    app.run()