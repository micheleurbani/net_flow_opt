"""Initialize Flask app."""
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache


db = SQLAlchemy()
login_manager = LoginManager()
cache = Cache()


def init_app():
    """Construct core Flask application."""
    app = Flask(__name__, instance_relative_config=False)
    # Application configurations
    app.config.from_object('config.Config')

    # Initialize plugins
    db.init_app(app)
    login_manager.init_app(app)
    cache.init_app(app)

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes
        from . import auth

        # Register Blueprints
        app.register_blueprint(routes.main_bp)
        app.register_blueprint(auth.auth_bp)

        db.create_all()  # Create sql tables for our data models
        cache.clear()    # Clear pre-existing cache

        # Import Dash application
        from .dashboard.dashboard import init_dashboard
        app = init_dashboard(app, cache)

        return app
