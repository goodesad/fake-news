from ml_logic.preprocessing import preprocess_csv_real_fakenews, preprocess_csv_news_notnews
from ml_logic.model import initialize_model_rf, initialize_model_nnn, \
                                            compile_model, \
                                                train_model_nnn, train_model_rf
from ml_logic.data_prep import \
                        prepare_data_news_notnews, prepare_pred_news_notnews, \
                        prepare_data_real_fakenews, prepare_pred_real_fakenews

from ml_logic.examples import *

import pandas as pd
import numpy as np
import os

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Flatten, Masking, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plan
# News / Not News
## 1. Take news articles and not news articles
## 2. Preprocess all articles
## 3. Train model to recognize news and not news articles


# Real / Fake News
## 1. Take raw csv (just News articles)
## 2. Preprocess X (the text column) (clean)
## 3. Prepare the data for model
### => Separate in X_train, X_test, y_train, y_test
## 4. Train the model to recognize real and fake news
## 5. Create predictor taking user input 'X_pred'

# TRAINING THE MODEL FROM LOCAL DATA
def preprocess_and_train_news_notnews():

    data = preprocess_csv_news_notnews()

    X_proc = data['text']
    y = data['target']

    X_train, X_test, y_train, y_test = prepare_data_news_notnews(X_proc, y)

    model = train_model_nnn(
                compile_model(
                    initialize_model_nnn()
                    ),
                X_train, y_train)

    return model

# PREDICTING NEWS OR NOT NEWS FROM USER INPUT
def news_notnews(X_pred):

    # loading model instead of calling from previous function to avoid:
    # "history has no function .predict" error

    model = tensorflow.keras.models.load_model('saved/models/model_news_notnews.tf')

    pred_data = prepare_pred_news_notnews(X_pred)

    result = model.predict(pred_data)[0][0]

    if result>0.5:
        return 1

    else:
        return 0


def preprocess_and_train_real_fakenews():

    cleanfake = pd.read_csv('data/clean_data_Fake.csv')
    cleantrue = pd.read_csv('data/clean_data_True.csv')

    cleantrue['target'] = 1

    data = pd.concat ([cleanfake, cleantrue], ignore_index=True)
    data = data.sample(n=len(data), ignore_index=True)

    X_proc = data['text']
    y = data['target']

    X_train, X_test, y_train, y_test = prepare_data_real_fakenews(X_proc, y)

    model = train_model_rf(
                compile_model(
                    initialize_model_rf()),
                X_train, y_train)

    return model

def real_fakenews(X_pred):

    pred_data = prepare_pred_real_fakenews(X_pred)

    model = tensorflow.keras.models.load_model('saved/models/model_real_fakenews.tf')
    result = model.predict(pred_data)[0][0]
    print (result)

    # 1 = true
    if result > 0.5:
        return 1, result
    # 0 = fake
    else:
        return 0, result

def main(X_pred, train = False):

    if train:
        # preprocess data for news / not news analysis
        model_news_notnews = preprocess_and_train_news_notnews()
        # preprocess data for real / fake news analysis
        model_real_fakenews = preprocess_and_train_real_fakenews()
        # get the result

    news_not_news_result = news_notnews(X_pred)

    # If article is news (RESULT = 1), loops into real or fake news analysis
    if news_not_news_result == 1:
        # run real/fake news analysis, keep result and pred value
        real_fakenews_result, pred_value = real_fakenews(X_pred)

        # result = 1 => real news
        if real_fakenews_result == 1:
            print ('RF Predicted value: ', pred_value, ' \n This is real news!')
        # result = 0 => fake news
        else:
            print ('RF Predicted value: ', pred_value, ' \n This is fake news!')

    # result = 0 => not news
    else:
        print ('This is not news!')



main(test_not_news)
