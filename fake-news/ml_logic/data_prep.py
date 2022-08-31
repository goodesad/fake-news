
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


def prepare_data_news_notnews(X,y):

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.25)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_raw)
    X_train_token = tokenizer.texts_to_sequences(X_train_raw)
    X_test_token = tokenizer.texts_to_sequences(X_test_raw)
    X_train = pad_sequences(X_train_token, dtype='int32',
                            padding='post', maxlen=300)

    X_test = pad_sequences(X_test_token, dtype='int32',
                           padding='post', maxlen=300)

    # save fitted tokenizer as pickle file
    with open ('saved/pickles/news_notnews_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, X_test, y_train, y_test

def prepare_pred_news_notnews(text):

    # load tokenizer pickle

    with open('saved/pickles/news_notnews_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    pred_token = tokenizer.texts_to_sequences([text])
    pred_pad = pad_sequences(pred_token,
                             padding='post',
                             maxlen=300,
                             dtype='int32')

    return pred_pad



def prepare_data_real_fakenews(X,y):

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.25)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_raw)
    X_train_token = tokenizer.texts_to_sequences(X_train_raw)
    X_test_token = tokenizer.texts_to_sequences(X_test_raw)
    X_train = pad_sequences(X_train_token, dtype='int32', padding='post', maxlen=300)
    X_test = pad_sequences(X_test_token, dtype='int32', padding='post', maxlen=300)

    # save fitted tokenizer as pickle file
    with open ('saved/pickles/real_fakenews_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, X_test, y_train, y_test

def prepare_pred_real_fakenews(text):

    # load tokenizer pickle

    with open('saved/pickles/real_fakenews_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    pred_token = tokenizer.texts_to_sequences([text])
    pred_pad = pad_sequences(pred_token,
                             padding='post',
                             maxlen=300,
                             dtype='int32')

    return pred_pad
