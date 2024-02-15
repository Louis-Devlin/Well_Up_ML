
from flask import Flask
from well_up_ml.db.db import db
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)


    with app.app_context():
        from well_up_ml.routes import sentiment 
        app.register_blueprint(sentiment.sentiment_bp)

    return app