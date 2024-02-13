from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Sentiment(db.Model): 
    id = db.Column(db.Integer, primary_key=True)
    sentiment_text = db.Column(db.String(250), nullable=False)
    sentiment_label = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "sentiment_text": self.sentiment_text,
            "sentiment_label": self.sentiment_label
        }