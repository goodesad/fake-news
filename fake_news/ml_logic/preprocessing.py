import pandas as pd
import os
import string
from nltk.stem import WordNetLemmatizer as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')

punctuations = string.punctuation + '’' + '“' + '‘' + '”'

def lemmatize (words):
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
        sentence = sentence.strip()

        sentence = sentence.lower()

        sentence = ''.join(char for char in sentence if not char.isdigit())

        for punctuation in punctuations:
            sentence = sentence.replace(punctuation, ' ')

        sentence = sentence.strip()

        sentence = word_tokenize (sentence)

        final = [word for word in sentence if word not in stop_words and len(word)>=3]

        lemmatize(final)

        return ' '.join(final)
    
    
def preprocess_csv_news_notnews():

    # not news
    lyricsdf = pd.read_csv('fake_news/raw_data/song_lyrics.csv')
    recipesdf = pd.read_csv('fake_news/raw_data/recipes.csv')
    fooddf = pd.read_csv('fake_news/raw_data/food_reviews.csv')
    bookdf = pd.read_csv('fake_news/raw_data/book_descriptions.csv')

    # news
    cleanfake = pd.read_csv('fake_news/raw_data/clean_data_Fake.csv')
    cleantrue = pd.read_csv('fake_news/raw_data/clean_data_True.csv')

    # preprocessing nonnews dataframes
    food_proc = fooddf['Text'].apply(preprocessing)
    book_proc = bookdf['description'].apply(preprocessing)
    recipes_proc = recipesdf['full'].apply(preprocessing)
    lyrics_proc = lyricsdf['Lyric'].apply(preprocessing)

    # making complete non news dataframe
    samples = 13575
    notnewsdf = pd.DataFrame(pd.concat([book_proc.sample(samples),
                                        food_proc.sample(samples),
                                        recipes_proc.sample(samples),
                                        lyrics_proc],
                                       ignore_index=True), columns=['text'])

    notnewsdf = notnewsdf.sample(n=len(notnewsdf), ignore_index=True)

    notnewsdf['target'] = 0


    # making complete news dataframe and sampling to randomize order
    newsdf = pd.concat([cleanfake, cleantrue], ignore_index=True)
    newsdf = newsdf.sample(n=len(newsdf), ignore_index=True)

    newsdf['target'] = 1

    # making full df (90 000 rows)
    fulldf = pd.concat([newsdf, notnewsdf], ignore_index=True)
    fulldf = fulldf.sample(n=len(fulldf), ignore_index=True)

    return fulldf


##############################################################################


# not being used because data already cleaned but keeping for possible later use
def preprocess_csv_real_fakenews(X: pd.DataFrame):

    X_proc = preprocessing(X)

    return X_proc
