import os, pickle
from cleaning import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenizing(text,turn):
    if turn == 1:
        with open(os.path.join('tokenizers', 'notnewsfromnews.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
            
    if turn == 2:
        with open(os.path.join('tokenizers', 'fakefromtrue.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
            
    text_token = tokenizer.texts_to_sequences([text])
    text_token = pad_sequences(text_token, dtype='int32', padding='post', maxlen=300)
    
    return text_token
    

def predicting(text):
    text = preprocessing(text)
    
    turn = 1
    text = tokenizing(text,turn)
    model = load_model(os.path.join('models', 'notnewsfromnews.tf'))
    
    if model.predict (text) > 0.5:
        turn = 2
        text = tokenizing(text,turn)
        model = load_model(os.path.join('models', 'fakefromtrue.tf'))
        
        if model.predict (text)> 0.5:
            return 'This news is true'
        return 'This news is fake'
    return 'This is not news'

