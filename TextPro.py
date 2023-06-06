from collections import defaultdict

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



documents = []
cleanDoc=[]

def text_pr(list):
    for d in list:
        if d !='\n':
         x = remove_punctuation(d)
         x = text_lowercase(x)
         x = remove_numbers(x)
         x1 = remove_stopwords(x)
         x2 = stem_words(x1)
         x2 = lemmatize_word(x2)
         cleanDoc.append(x2)
    return cleanDoc

def date(doc):
    reg = "?<Time>^(?:0?[1-9]:[0-5]|1(?=[012])\d:[0-5])\d(?:[ap]m)?"
    reg1 = "^([0]\d|[1][0-2])\/([0-2]\d|[3][0-1])\/([2][01]|[1][6-9])\d{2}(\s([0-1]\d|[2][0-3])(\:[0-5]\d){1,2})?$"
    reg2 = "^([1-9]|1[0-2]|0[1-9]){1}(:[0-5][0-9][aApP][mM]){1}$"
    reg3 = "^(\d{4})-((0[1-9])|(1[0-2]))-(0[1-9]|[12][0-9]|3[01])$"
    reg4 = "\b(0[0-9]|1[0-9]|2[0-3])(\:)(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])(\:)(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])\b"
    reg5 = "^(1[0-2]|0?[1-9]):([0-5]?[0-9])( AM| PM)$"
    reg6 = "((\(\d{2}\) ?)|(\d{2}/))?\d{2}/\d{4} ([0-2][0-9]\:[0-6][0-9])"
    for i in doc:
        dat = re.match("reg" or "reg1" or "reg2" or "reg3" or "reg4" or "reg5" or "reg6", listToString(i))
    print(dat)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

    # test = remove_punctuation()
    # print(test)


def text_lowercase(text):
    return text.lower()


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result



def remove_whitespace(text):
    return (" ".join(text.split()))


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


stemmer = PorterStemmer()


def stem_words(text):
    stems = [stemmer.stem(word) for word in text]
    return stems


lemmatizer = WordNetLemmatizer()


def lemmatize_word(text):
    lemmas = [lemmatizer.lemmatize(word, pos='n') for word in text]
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in lemmas]
    lemmas = [lemmatizer.lemmatize(word, pos='a') for word in lemmas]
    return lemmas


def listToString(s):
    str1 = " "
    return (str1.join(s))