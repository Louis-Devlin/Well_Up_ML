from flask_sqlalchemy import SQLAlchemy

from well_up_ml.db.db import db

class Sentiment(db.Model): 
    SentimentID = db.Column(db.Integer, primary_key=True)
    sentiment_text = db.Column(db.String(250), nullable=False)
    sentiment_label = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            "SentimentID": self.SentimentID,
            "sentiment_text": self.sentiment_text,
            "sentiment_label": self.sentiment_label
        }