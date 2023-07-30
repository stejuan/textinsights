import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from gensim.models import Word2Vec
import numpy as np
import pickle
import top2vec as Top2Vec
from top2vec import Top2Vec
from nltk.tokenize import sent_tokenize
import gensim



def topic_model(documents):

    # Preprocess the data
    stopwords = gensim.parsing.preprocessing.STOPWORDS

    texts = []
    for doc in documents:
        tokens = gensim.utils.simple_preprocess(doc)
        tokens = [token for token in tokens if token not in stopwords]
        texts.append(tokens)
    count = 0
    for l in texts:
        if len(l)!=0:
            count+=1
    if count==0:
        return "None"
    # Create a dictionary from the texts

    dictionary = gensim.corpora.Dictionary(texts)

    # Create a corpus from the texts
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the model
    model = gensim.models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)

    # Print the topics
    # for topic in model.print_topics():
    #    print(topic)
    
    return model

'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
tokens = sent_tokenize('my name Nathan Nguyen. I a the best in the world.')
mod = topic_model(tokens)
fig, ax = plt.subplots(5, 1)

# loop through the topics and generate a wordcloud for each one
for i, topic in enumerate(mod.print_topics()):
    # get the words for the topic
    topic_words = [word for word, _ in mod.show_topic(topic[0])]
    topic_text = " ".join(topic_words)
    
    # generate the wordcloud
    wordcloud = WordCloud().generate(topic_text)

    # add the wordcloud to the subplot
    ax[i].imshow(wordcloud, interpolation='bilinear')
    ax[i].axis("off")
plt.savefig(f'assets/wordcloud.png',dpi = 300)

'''
l = []
l.append(45)