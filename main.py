import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash import html
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
import xgboost as xgb
import re
import pickle
import gensim
from topic_model import topic_model
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import random

stop_words = set(stopwords.words('english'))

with open("vmodel.pkl", "rb") as file:
    vectorizer = pickle.load(file)

from summarizer_model import summarize

with open("xgbmodel.pkl", "rb") as file:
    xgbmodel = pickle.load(file)

card1 = dbc.Card([
    html.H2(children="Input/Paste product review below:"),
    dbc.Input(id="input", placeholder="Enter Text", type="text"),
    dbc.Button("Submit", id="example-button", className="me-1",n_clicks=0)
    ],
    body=True,)


card2 = dbc.Card([
    html.H2(children="Sentiment Prediction:"),
    html.P(id="output")
    ],
    body=True,)

card3 = dbc.Card([
    html.H2(children="Text Summarization:"),
    html.P(id="output2")
    ],
    body=True,)

card4 = dbc.Card([
    html.H2(children="Topic Model:"),
    html.Div(id='wordclouds')
    ],
    body=True,)



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
     html.H1(children='TextInsights',style={'textAlign':'center'}),
     html.H3(children='By: Steven Chung',style={'textAlign':'center'}),
     html.P(children="This project utilizes machine learning to provide insights on user input (mainly product reviews). Enter text input below.",style={'textAlign':'center'}),
     
     dbc.Col([card1]),
     dbc.Col([
        dbc.Row([card2]),
        dbc.Row([card3]),
        dbc.Row([card4])
     ]),
    ],fluid=True,)

@app.callback([Output("output", "children"),Output("output2", "children"),Output("wordclouds", "children")], [Input("example-button", "n_clicks")],[State("input", "value")])
def outputs(n_clicks,value):
    if n_clicks == 0:
        return "No Text Entered",'No Text Entered',"No Text Entered"
    else:
        # Sentiment Analysis
        stripped = re.sub('[^A-Za-z ]+', '', value)
        tokens = word_tokenize(stripped)
        tokens = [w for w in tokens if w.lower() not in stop_words]
        tokens = ' '.join(tokens)
        inn = vectorizer.transform([tokens])
        value1 = xgbmodel.predict(inn)
        if value1 == 1:
            value1 = "Negative Sentiment"
        else:
            value1 = "Positive Sentiment"

        # Text Summarization
        value2 = summarize(value)

        # Topic Model
        
        tokens = sent_tokenize(value)
        mod = topic_model(tokens)

        if mod != "None":
            # loop through the topics and generate a wordcloud for each one
            for i, topic in enumerate(mod.print_topics()):
                # get the words for the topic
                topic_words = [word for word, _ in mod.show_topic(topic[0])]
                topic_text = " ".join(topic_words)
                
                # generate the wordcloud
                wordcloud = WordCloud().generate(topic_text)
        
                # add the wordcloud to the subplot
            wordcloud.to_file("assets/wordcloud2.png")
            value3 = html.Img(src=app.get_asset_url('wordcloud2.png')+ "?t={}".format(random.random()),style={"height": "800px", "width": "100%"})
            return value1,value2,value3
        return value1,value2,"None"


if __name__ == '__main__':
    app.run_server(debug=False)
