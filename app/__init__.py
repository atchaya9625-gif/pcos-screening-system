import os
from flask import Flask
from app.models import db


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'change-this-in-production'

    # SQLite database - stored as a file inside the project folder
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(basedir, "pcos_history.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    from app.routes import main
    app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app
