from importlib.resources import path
import io
import os
import random
import re
import sys
import xml.etree.ElementTree

import numpy as np
import pandas as pd
import os
import tweepy as tw
from textblob import TextBlob

from sklearn.model_selection import train_test_split

import preprocessor as p

from gensim.parsing.preprocessing import remove_stopwords

import yaml
import csv
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))["prepare"]

# if len(sys.argv) != 2:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython prepare.py data-file\n")
#     sys.exit(1)

# Test data set split ratio
split = params["split"]
#random.seed(params["seed"])

parent_dir ="D:/BigData2/GitRepo/BigData2_project"
directory = "Data/prepared"
data_path = os.path.join(parent_dir,directory)
os.makedirs(data_path, exist_ok=True)

data_input = os.path.join(sys.argv[1], "Sentiment_Analysis_Log_reg.csv")

#input = pd.read_csv('D://BigData2/GitRepo/BigData2_project/Data/Ukraine_Data.csv')

# output_train = os.path.join("data", "prepared", "train.csv")
# output_test = os.path.join("data", "prepared", "test.csv")


# Preprocess the data:

#####################################################################################################

df = pd.DataFrame(input)

final_input = df['Tweets']

final_input = final_input.dropna()
final_input = final_input.drop_duplicates()

def preprocess_tweet(row):
    #text = row['Tweets']
    text = row[0]
    text = p.clean(text)
    return text

final_input['text'] = final_input.apply(preprocess_tweet)

def stopword_removal(row):
    # text = row['text']
    text = row[0]
    text = remove_stopwords(text)
    return text

final_input['text'] = final_input.apply(stopword_removal)


#final_input['text'] = final_input['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+',' ')


#####################################################################################################


# Split the dataset using the parameters and save as csv files:

df_train, df_test = train_test_split(final_input, test_size = split, random_state=random.seed(params["seed"]))

test_data_path = os.path.join(data_path, "test_data.csv")
train_data_path = os.path.join(data_path, "train_data.csv")

# output_dir = Path('D:/BigData2/GitRepo/BigData2_project/Data/prepared')
# output_dir.mkdir(parents=True, exist_ok=True)
# df.to_csv(output_dir / train_data)
# df.to_csv(output_dir / test_data)

# with open('data_path','w',encoding='utf-8') as csvFile: #EDIT - because comment.
#     writer = csv.writer(csvFile)
#     writer.writerows(df_train)
#     writer.writerows(df_test)

# df_train.to_csv(data_path,"train_data.csv")

# df_test.to_csv(data_path,"test_data.csv")

df_train.to_csv(train_data_path)
df_test.to_csv(test_data_path)


# df_train.to_csv(os.path.join("Data","prepared", "train.csv"))

# df_test.to_csv(os.path.join("Data","prepared", "test.csv"))

# df_train.to_csv(os.path.join("data/prepared",'/train.csv'), encoding="utf8")

# df_test.to_csv(os.path.join("data/prepared",'/train.csv'), encoding="utf8")


# df_train.to_csv('C:\\BDBA\\train.csv')

# df_test.to_csv('C:\\BDBA\\test.csv')










# def process_posts(fd_in, fd_out_train, fd_out_test, target_tag):
#     num = 1
#     for line in fd_in:
#         try:
#             fd_out = fd_out_train if random.random() > split else fd_out_test
#             attr = xml.etree.ElementTree.fromstring(line).attrib

#             pid = attr.get("Id", "")
#             label = 1 if target_tag in attr.get("Tags", "") else 0
#             title = re.sub(r"\s+", " ", attr.get("Title", "")).strip()
#             body = re.sub(r"\s+", " ", attr.get("Body", "")).strip()
#             text = title + " " + body

#             fd_out.write("{}\t{}\t{}\n".format(pid, label, text))

#             num += 1
#         except Exception as ex:
#             sys.stderr.write(f"Skipping the broken line {num}: {ex}\n")


# os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

# with io.open(final_input, encoding="utf8") as fd_in:
#     with io.open(output_train, "w", encoding="utf8") as fd_out_train:
#         with io.open(output_test, "w", encoding="utf8") as fd_out_test:
#             process_posts(fd_in, fd_out_train, fd_out_test, "<r>")
