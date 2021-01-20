"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.html",
        title="Plotly Dash integration in Flask",
        description="Embed Plotly Dash into a Flask application.",
        template="home-template",
        body="This is a homepage served with Flask.",
    )

@app.route("/login")
def login():
    """Return login page."""
    return render_template(
        "login.html",
        template="login-form"
    )