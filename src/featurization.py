import os
import pickle
import sys
from textblob import TextBlob
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import random
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import re
import string

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

input = sys.argv[1]

data_input = os.path.join(input,"extracted_data.csv")


data = pd.read_csv(data_input)

text = data['Tweets']

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create two new columns
data['Subjectivity'] = data['Tweets'].apply(getSubjectivity)
data['Polarity'] = data['Tweets'].apply(getPolarity)


# Function to compute the negative, neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

data['sentiment'] = data['Polarity'].apply(getAnalysis)

data = data.replace('Positive', 1)
data = data.replace('Negative', -1)
data = data.replace('Neutral', 0)

# Features and Labels
data['clean_tweet'] = np.vectorize(remove_pattern)(data['Tweets'],"@[\w]*")
tokenized_tweet = data['clean_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['clean_tweet'] = tokenized_tweet
data['body_len'] = data['Tweets'].apply(lambda x:len(x) - x.count(" "))
data['punct'] = data['Tweets'].apply(lambda x:count_punct(x))


data_path = os.path.join("data", "features")

os.makedirs(data_path, exist_ok=True)

data.to_csv(os.path.join(data_path, "features.csv"), index=None)