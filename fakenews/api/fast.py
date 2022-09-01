
from pyexpat import model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fakenews.functions import preprocess_txt
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import tensorflow as tf
from tensorflow import keras

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# criando um endpoint predict pro request do usuário (txt)

@app.get("/predict")
def predict(txt: str):


    # load dos modelos news/notnews e fake/true

    # lembrar de salvar o modelo nesse caminho

    model_news_notnews = tf.keras.models.load_model('fakenews/models/notnewsfromnews.tf')

    model_fakefromtrue = tf.keras.models.load_model('fakenews/models/fakefromtrue.tf')


    # preprocessamento do texto do usuário


    # txt = txt.replace ("’", "'")
    # txt = txt.replace ("‘", "'")
    # txt = txt.replace ('“', '"')
    # txt = txt.replace ('”', '"')

    # X = pd.Series(txt)

    # X_processed = X.apply(preprocess_txt)

    X_processed = preprocess_txt(txt)

    # tokenizing for news/not news

    with open('fakenews/tokenizers/notnewsfromnews.pickle', 'rb') as handle:
        tokenizer_news_notnews = pickle.load(handle)

    X_token_news_notnews = tokenizer_news_notnews.texts_to_sequences([X_processed])

    X_pad_news_notnews = pad_sequences(X_token_news_notnews, dtype='int32', padding='post', maxlen=300)

    y_news_pred = model_news_notnews.predict(X_pad_news_notnews.reshape(1,300))

    if y_news_pred[0][0] < 0.5:
        return {"Class": "Not News"}

    else:

    # tokenizing for fake/true

        with open('fakenews/tokenizers/fakefromtrue.pickle', 'rb') as handle:
            tokenizer_fakefromtrue = pickle.load(handle)

        X_token_fakefromtrue = tokenizer_fakefromtrue.texts_to_sequences([X_processed])

        X_pad_fakefromtrue = pad_sequences(X_token_fakefromtrue, dtype='int32', padding='post', maxlen=300)

        y_fakefromtrue_pred = model_fakefromtrue.predict(X_pad_fakefromtrue.reshape(1,300))

    # returning final score

        return {"Class": str(y_fakefromtrue_pred[0][0])}
