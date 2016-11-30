from app import app, models
from .models import *
from flask import render_template, Flask, redirect, url_for, session, request, escape


@app.route("/")
@app.route("/index")
def index():
    '''
    Displays the home page
    '''
    return render_template("index.html")

@app.route("/check-for-abuse", methods=["POST"])
def check_for_abuse():
    '''
    Checks if entered text contains abusive context
    '''
    if request.method == "POST":
        print("Helloooo!")
        abusive_text = request.form["abusive-text"]
        clf = models.retrieve_model()
        # Code to check if text is abusive
        if clf == True:
            return render_template("index.html", abusive=True)
        else:
            return render_template("index.html", abusive=False)
