import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from gensim.models import Word2Vec
import xgboost as xgb
import numpy as np
import pickle

data = pd.read_csv('reviews.csv',header=None,names=["Sentiment", "Title", "Body"])
newdata = data[0:1000].copy() # Only use first 100,000 reviews to save computing/training time

for i in range(newdata.shape[0]): # Remove special characters
    newdata.iloc[i,1]=re.sub('[^A-Za-z ]+', '', str(newdata.iloc[i,1]))
    newdata.iloc[i,2]=re.sub('[^A-Za-z ]+', '', str(newdata.iloc[i,2]))
newdata["Tokenized_Title"] = [word_tokenize(line) for line in newdata['Title']] 
newdata["Tokenized_Body"] = [word_tokenize(line) for line in newdata['Body']] 
stop_words = set(stopwords.words('english'))

def remove_stop(s):
    return [w for w in s if not w.lower() in stop_words]

newdata["Tokenized_Title"] = [remove_stop(line) for line in newdata['Tokenized_Title']]
newdata["Tokenized_Body"] = [remove_stop(line) for line in newdata['Tokenized_Body']]

X = newdata["Tokenized_Body"]
y = newdata["Sentiment"]

Xx = newdata['Tokenized_Body'].apply(lambda x:' '.join(x))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=700)
vectorizer.fit(Xx)
features = vectorizer.transform(Xx)

tf_idf = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names())

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(tf_idf, y, test_size=0.2, random_state=13)




'''
# Method2: using Word2Vec (word embeddings)


model = Word2Vec(sentences=X_train.values, vector_size=1000, window=5, min_count=1, workers=4)

def getVectors(dataset):
  vectors=[]
  for dataItem in dataset:
    wordCount=0
    singleDataItemEmbedding=np.zeros(1000)
    for word in dataItem:
        try:
            singleDataItemEmbedding=singleDataItemEmbedding+model.wv[word]
            wordCount=wordCount+1
        except:
            pass
    singleDataItemEmbedding=singleDataItemEmbedding/wordCount 
    vectors.append(singleDataItemEmbedding)
  return np.mean(np.nan_to_num(vectors),axis=0)

X_train=X_train.apply(getVectors)
X_train = np.array(X_train.to_list())
X_test=X_test.apply(getVectors)
X_test = np.array(X_test.to_list())
'''

xgbmodel = xgb.XGBClassifier()
xgbmodel.fit(X_train,y_train)
preds = xgbmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,preds)) # XGB



with open('xgbmodel.pkl', 'wb') as file:
    pickle.dump(xgbmodel, file)

with open('vmodel.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


