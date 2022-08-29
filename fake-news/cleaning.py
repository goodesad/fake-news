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


def preprocessing(sentence):

    sentence = sentence.str.replace ("’", "'", regex = False)
    sentence = sentence.str.replace ("‘", "'", regex = False)
    sentence = sentence.str.replace ('“', '"', regex = False)
    sentence = sentence.str.replace ('”', '"', regex = False)
    
    sentence = sentence.strip()
    sentence = sentence.lower()
    
    sentence = ''.join(char for char in sentence if not char.isdigit())
    
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, ' ') 
    
    sentence = sentence.strip()
    sentence = word_tokenize (sentence)
    
    final = []
    for word in sentence:
        if word in stop_words:
            sentence.remove(word)
        if len(word) >= 3:
            final.append(word)
            
    lematize(final)
            
    return ' '.join(final)