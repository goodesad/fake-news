
from pyexpat import model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from fakenews.functions import preprocess_txt
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy deep-learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the uvicorn server starts
# Then to store the model in an `app.state.model` global variable accessible across all routes!
# This will prove very useful for demo days
# app.state.model = load_model()
# $WIPE_END


@app.get("/predict")
def predict(txt: str): # how do I establish a connection to this variable inputed in the website project?
    print(os.getcwd())
    #X_pred = {txt:[txt]}
    model = joblib.load('fakenews/api/model_cnn_lucas.joblib')
    X = pd.Series(txt)

    X = X.str.replace ("’", "'", regex = False)
    X = X.str.replace ("‘", "'", regex = False)
    X = X.str.replace ('“', '"', regex = False)
    X = X.str.replace ('”', '"', regex = False)

    X_processed = X.apply(preprocess_txt)

    with open('fakenews/api/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    X_token = tokenizer.texts_to_sequences(X_processed)

    X_pad = pad_sequences(X_token, dtype='int32', padding='post', maxlen=300)

    y_pred = model.predict(X_pad.reshape(1,300))

    if y_pred[0][0] > 0.5:
        return {"Class": 'True'}
    else:
        return {"Class": 'False'}
