from ml_logic.preprocessing import preprocess_csv_real_fakenews, preprocess_csv_news_notnews
from ml_logic.model import initialize_model_rf, initialize_model_nnn, \
                                            compile_model, \
                                                train_model_nnn, train_model_rf
from ml_logic.data_prep import \
                        prepare_data_news_notnews, prepare_pred_news_notnews, \
                        prepare_data_real_fakenews, prepare_pred_real_fakenews
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


def news_notnews(X_pred):

    # loading model instead of calling from previous function to avoid:
    # "history has no function .predict" error

    model = tensorflow.keras.models.load_model('saved/models/model_news_notnews.tf')

    pred_data = prepare_pred_news_notnews(X_pred)

    result = model.predict(pred_data)[0]

    if result>0.5:
        return 1

    else:
        return 0



def preprocess_and_train_real_fakenews():

    cleanfake = pd.read_csv('data/clean_data_Fake.csv')
    cleantrue = pd.read_csv('data/clean_data_True.csv')


    data = pd.concat ([cleanfake, cleantrue], ignore_index=True)
    data = data.sample(n=len(data), ignore_index=True)

    X_proc = preprocess_csv_real_fakenews(data['text'])
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
    result = model.predict(pred_data)[0]

    # 1 = true
    if result > 0.5:
        return 1
    # 0 = fake
    else:
        return 0


def main(X_pred):
    # preprocess data for news / not news analysis
    model_news_notnews = preprocess_and_train_news_notnews()
    # get the result
    news_not_news_result = news_notnews(X_pred)

    # result = 1 => news
    if news_not_news_result == 1:
        # preprocess data for real / fake news analysis
        model_real_fakenews = preprocess_and_train_real_fakenews()
        # get the result
        real_fakenews_result = real_fakenews(X_pred)

######
###### EDIT: Removed model parameter from predict functions news_notnews()
###### and real_fakenews()
###### => because model is now being fitted, saved and loaded
######      => doing model = model.fit(...) was returning a history which cannot be saved or used for .predict
######



        # result = 1 => real news
        if real_fakenews_result == 1:
            print ('That is real news!')
        # result = 0 => fake news
        else:
            print ('That is fake news!')

    # result = 0 => not news
    else:
        print ('not news')


test = """A number of Western nations have no desire to see an end to the Ukraine conflict and are also taking steps to derail a UN- and Turkey-brokered grain deal signed by Moscow and Kiev, the latter’s foreign minister told local media on Tuesday.

Speaking to Turkish-language outlet Haber Global, Foreign Minister Mevlut Cavusoglu stated that several Western countries “want the war to continue,” adding that it is not only the US, but also a handful of NATO members.

“There were also those who wanted to sabotage the grain deal,” he noted, adding that the US had nothing to do with these efforts and is in fact being helpful.


“The US contribution was as follows: the removal of export barriers for Russian fertilizers, unblocking ports, [lifting restrictions on] banking transactions, etc. But some countries from Europe wanted to sabotage it,” he said, signaling that Turkey continues to work to make sure the grain deal is upheld.
The agreement to unblock grain exports via the Black Sea was signed by Moscow and Russia at UN-brokered talks in Istanbul in late July, and aims to maintain safe transit routes. The agreement is also supposed to allow Russia to deliver fertilizers and food products to global markets. Many experts and officials deem the agreement to be instrumental in alleviating global food security issues.

Wheat deliveries from Ukraine, a major producer, were disrupted after Russia launched its military operation in the neighboring state in late February. The sides blamed each other for causing the crisis.

Last week, Russian President Vladimir Putin accused the US of attempting to prolong the Ukraine conflict by “pumping the Kiev regime with weapons, including heavy weapons.”

Putin stated that the Ukrainians have been assigned the role of “cannon fodder” in Washington’s “anti-Russia project.” The president also said that Moscow launched its offensive in Ukraine to “ensure the security of Russia and its citizens, and defend the people of Donbass from genocide.”

US President Joe Biden said in June that NATO will support Ukraine “as long as it takes” to make sure Kiev is not defeated."""



main(test)
