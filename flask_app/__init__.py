"""Initialize Flask app."""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.Config')

    db.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        db.create_all() # Create sql tables for our data models

        # Import Dash application
        from .dashboard.dashboard import init_dashboard
        app = init_dashboard(app)

        return app