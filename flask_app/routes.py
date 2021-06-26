"""Routes for parent Flask app."""
from flask import current_app as app
from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   send_from_directory)


# Blueprint Configuration
main_bp = Blueprint(
    'main_bp', __name__,
    template_folder='templates',
    static_folder='static'
)


@main_bp.route('/', methods=['GET'])
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


@main_bp.route("/logout")
def logout():
    """User log-out logic."""
    logout_user()
    return redirect(url_for('auth_bp.home'))


@main_bp.route("/dashapp")
def dashapp():
    return redirect(url_for('/dashapp/'))


@main_bp.route("/dashapp/download/<path:filename>")
def download(filename):
    return send_from_directory(
        main_bp.static_folder,
        filename,
        as_attachment=True,
    )
