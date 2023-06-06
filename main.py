import csv
import glob
import os
import stat
import string

import ir_datasets
import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import TextPro
from collections import defaultdict, OrderedDict

dataset1 = ir_datasets.load("antique/test")
datase2 = ir_datasets.load("antique/train")
datasettt=[]
tsv_file1 = open("C:\\Users\\96393\\.ir_datasets\\antique\\collection.tsv")

txt_file1 = open("docs/project_output.txt", "w")
data = csv.reader(tsv_file1, delimiter='\t')
for row in data:
    str = "\t".join(row)
    datasettt.append(str)
    txt_file1.writelines(str + '\n')
txt_file1.close

# tsv_file2 = open("C:\\Users\\96393\\.ir_datasets\\antique\\collection.tsv")
# txt_file2 = open("docs/project_output2.txt", "w")
# data=csv.reader(tsv_file2, delimiter='\t')
# for row in data:
#     str="\t".join(row)
#     txt_file2.writelines(str+'\n')
# txt_file2.close


txt_file1 = open("docs/project_output.txt", encoding='utf8')
print("text processing ..")
text1 = TextPro.text_pr(txt_file1)
print(text1)



def give_path(fld_path):  # give path of the folder containing all documents
    dic = {}
    file_names = glob.glob(fld_path)
    files_150 = file_names[0:10]
    for file in files_150:
        name = file.split('/')[-1]
        with open(file, 'r', errors='ignore') as f:
            data = f.read()
        dic[name] = data
    return dic


def wordList_removePuncs(doc_dict):
    stop = stopwords.words('english') + list(string.punctuation) + ['\n']
    wordList = []
    for doc in doc_dict.values():
        for word in word_tokenize(doc.lower().strip()):
            if not word in stop:
                wordList.append(word)
    return wordList


def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}

    for word in vocab:
        for doc_id, doc in doc_dict.items():
            tf_docs[doc_id][word] = doc.count(word)
    return tf_docs


def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in doc_dict.values():
            #             if word in doc.lower().split():
            if word in word_tokenize(doc.lower().strip()):
                frq = frq + 1
        df[word] = frq
    return df


def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        idf[word] = np.log2((length + 1) / doc_fre[word])
    return idf


def tfidf(vocab, tf, idf_scr, doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id, doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr


def vectorSpaceModel(query, doc_dict, tfidf_scr):
    query_vocab = []
    stop_words = set(stopwords.words("english"))
    for word in query.split():
        if word not in stop_words:
          if word not in query_vocab:
             query_vocab.append(word)

    query_wc = {}
    for word in query_vocab:
        query_wc[word] =query.lower().split().count(word)

    relevance_scores = {}
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab:
            score += query_wc[word] * tfidf_scr[doc_id][word]
        relevance_scores[doc_id] = score
    sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    top_5 = {k: sorted_value[k] for k in list(sorted_value)[:5]}
    doc1 = list(top_5.keys())[0]
    print(doc1)
    return top_5


path = "docs/*"
docs = give_path(path)  # returns a dictionary of all docs
M = len(docs) # number of files in dataset
w_List = wordList_removePuncs(docs)  # returns a list of tokenized words
print("word list:\n", w_List)
vocab = list(set(w_List))  # returns a list of unique words
print("vocabulary:\n", vocab)
tf_dict = termFrequencyInDoc(vocab, docs)  # returns term frequency
print("term frequency:\n", tf_dict)
df_dict = wordDocFre(vocab, docs)  # returns document frequencies
idf_dict = inverseDocFre(vocab, df_dict, M)  # returns idf scores
tf_idf = tfidf(vocab, tf_dict, idf_dict, docs)  # returns tf-idf socres
print("tf-idf:\n", tf_idf)


from sklearn.cluster import KMeans
# Vectorize the dataset
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(datasettt)

# Define the number of clusters
k =3

# Create a k-means model and fit it to the data
km = KMeans(n_clusters=k)
km.fit(X)

# Predict the clusters for each document
y_pred = km.predict(X)

# Print the cluster assignments
print(y_pred)
# Output: [1 1 0 1 0]

import seaborn as sns

# convert to a sparse matrix
X = X.toarray()

# Create a scatter plot of the data colored by the predicted clusters
sns.scatterplot(x=X[:, 0], y=X[:, 5], hue=y_pred)
plt.show()


# Initialize and fit the Model
model = LogisticRegression()
model.fit(X, y_pred)

# Make prediction on the test set
pred = model.predict(X)

# calculating precision and reall
precision = precision_score(y_pred, pred, average=None)
recall = recall_score(y_pred, pred, average=None)
txt_file1.close()
print('Precision: ', precision)
print('Recall: ', recall)
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/handle_get', methods=['GET'])
def handle_get():
    query1=request.args['nm']
    top_doc1 = vectorSpaceModel(query1, docs, tf_idf)
    print('Top Documents for Query 1: \n', top_doc1)
    doc1 = list(top_doc1.keys())[0]
    f = open(doc1, 'r')
    file_contents = f.read()
    return file_contents

if __name__ == "__main__":
    app.run(debug=True)


