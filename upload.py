from flask import Flask
import os

UPLOAD_FOLDER = "C:/Users/Sachin/PycharmProjects/Flask-app/static/uploads"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024