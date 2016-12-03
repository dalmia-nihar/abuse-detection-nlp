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
        abusive_text = request.form["abusive-text"]
        confidence = models.is_abuse(abusive_text)
        # Code to check if text is abusive
        if confidence >= 0.5:
            return render_template("index.html", abusive=False)
        else:
            return render_template("index.html", abusive=True)


@app.route("/analyze-twitter")
def display_twitter_page():
    '''
    Opens the page for the user to enter the twitter handle
    '''
    return render_template("twitterAnalysis.html")


@app.route("/check-twitter-handle", methods=["POST"])
def check_twitter_handle():
    '''
    Performs analysis on the input twitter handle
    '''
    if request.method == "POST":
        twitter_handle = request.form["twitter-handle"]
        clf = models.retrieve_model()

        # Code to perform tweet analysis
        api = models.init_twitter_api()
        total_tweets = []
        for page in range(1, 20):
            tweets = api.user_timeline(twitter_handle, count=200, page=page)
            for tweet in tweets:
                total_tweets.append(tweet.text.encode('utf-8'))
        print(len(total_tweets))

        if clf == True:
            return render_template("twitterAnalysis.html", abusive=True)
        else:
            return render_template("twitterAnalysis.html", abusive=False)
