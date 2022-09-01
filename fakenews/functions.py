import string
from nltk.stem import WordNetLemmatizer as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = stopwords.words('english')

def lematize (words):
        for index, word in enumerate (words):
            words[index] = wn().lemmatize(word, pos='v')
        for index, word in enumerate (words):
            words[index] = wn().lemmatize(word, pos='r')
        for index, word in enumerate (words):
            words[index] = wn().lemmatize(word, pos='a')
        for index, word in enumerate (words):
            words[index] = wn().lemmatize(word, pos='n')
        for index, word in enumerate (words):
            words[index] = wn().lemmatize(word, pos='s')
        return ' '.join(words)


def preprocess_txt(txt):

    txt = txt.strip()

    txt = txt.lower()

    txt = ''.join(char for char in txt if not char.isdigit())

    # for punctuation in string.punctuation:
    #     txt = txt.replace(punctuation, ' ')

    for punctuation in (string.punctuation + "’" + "‘" + '“' + '”'):
        txt = txt.replace(punctuation, ' ')

    txt = txt.strip()

    txt = word_tokenize (txt)
    # for word in txt:
    #     if word in stop_words:
    #         txt.remove(word)

    # new_txt = []
    # for word in txt:
    #     if len(word) >= 3:
    #         new_txt.append(word)

    new_txt = [word for word in txt if word not in stop_words and len(word)>=3]

    lematize(new_txt)

    return ' '.join(new_txt)
