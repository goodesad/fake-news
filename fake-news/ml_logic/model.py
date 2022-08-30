from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Flatten, Masking, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# defining metrics
binacc = BinaryAccuracy()
prec = Precision()
rec = Recall()

##############################################

# rf = real / fake news model
def initialize_model_rf():
    with open('saved/pickles/real_fakenews_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = Sequential()
    model.add(Embedding(
        input_dim=(len(tokenizer.word_index))+1, # +1 for the 0 padding
        input_length=300, # Max_sentence_length (optional, for model summary)
        output_dim=100,
        mask_zero=True, # Built-in masking layer
    ))
    model.add(Conv1D(12, kernel_size=5, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model

#############

# nnn = news / not news model
def initialize_model_nnn():
    with open('saved/pickles/news_notnews_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    model = Sequential()
    model.add(Embedding(
        input_dim=(len(tokenizer.word_index))+1, # +1 for the 0 padding
        input_length=300, # Max_sentence_length (optional, for model summary)
        output_dim=100,
        mask_zero=True, # Built-in masking layer
    ))
    model.add(Conv1D(12, kernel_size=5, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    return model

##################################################


def compile_model(model):
    model.compile(loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.0001),
    metrics=[binacc, prec, rec])

    return model

##################################################

def train_model_nnn(model, X_train, y_train):
    es = EarlyStopping(monitor='val_binary_accuracy',patience=4,
                       restore_best_weights=True)

##### CHANGE BATCH_SIZE BACK TO 32 AND EPOCH TO 3
    model_final = model.fit(X_train, y_train,
                            epochs = 1,
                            callbacks=[es],
                            batch_size=2048,
                            verbose=1,
                            use_multiprocessing=True)

    model.save('saved/models/model_news_notnews.tf')

#########


def train_model_rf(model, X_train, y_train):
    es = EarlyStopping(monitor='val_binary_accuracy',patience=4,
                       restore_best_weights=True)

    model_final = model.fit(X_train, y_train,
                            epochs = 1,
                            callbacks=[es],
                            batch_size=2048,
                            verbose=1,
                            use_multiprocessing=True)

    model.save('saved/models/model_real_fakenews.tf')

    return model_final


    return model_final
