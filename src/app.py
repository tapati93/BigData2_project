import pickle
import numpy
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100


input = "data/features"

data_input = os.path.join(input,"features.csv")

data = pd.read_csv(data_input)

X = data['clean_tweet']
y = data['sentiment']
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

model = pickle.load(open("Logisticmodel.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        body_len = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([body_len,punct,vect],axis = 1)
        my_prediction = model.predict(total_data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)