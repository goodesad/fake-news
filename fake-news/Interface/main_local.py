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

    print (result)

    if result>0.5:
        return 1

    else:
        return 0


def preprocess_and_train_real_fakenews():

    cleanfake = pd.read_csv('data/clean_data_Fake.csv')
    cleantrue = pd.read_csv('data/clean_data_True.csv')


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

def main(X_pred):
    # preprocess data for news / not news analysis
    model_news_notnews = preprocess_and_train_news_notnews()
    # preprocess data for real / fake news analysis
    model_real_fakenews = preprocess_and_train_real_fakenews()
    # get the result

    news_not_news_result = news_notnews(X_pred)

    # If article is news (RESULT = 1), loops into real or fake news analysis
    if news_not_news_result == 1:
        # get the result
        real_fakenews_result, pred_value = real_fakenews(X_pred)

######
###### EDIT: Removed model parameter from predict functions news_notnews()
###### and real_fakenews()
###### => because model is now being fitted, saved and loaded
######      => doing model = model.fit(...) was returning a history which cannot be saved or used for .predict
######

        # result = 1 => real news
        if real_fakenews_result == 1:
            print ('Predicted value: ', pred_value, ' \n That is real news!')
        # result = 0 => fake news
        else:
            print ('Predicted value: ', pred_value, ' \n That is fake news!')

    # result = 0 => not news
    else:
        print ('not news')


test_fake = """A number of Western nations have no desire to see an end to the Ukraine conflict and are also taking steps to derail a UN- and Turkey-brokered grain deal signed by Moscow and Kiev, the latter’s foreign minister told local media on Tuesday.

Speaking to Turkish-language outlet Haber Global, Foreign Minister Mevlut Cavusoglu stated that several Western countries “want the war to continue,” adding that it is not only the US, but also a handful of NATO members.

“There were also those who wanted to sabotage the grain deal,” he noted, adding that the US had nothing to do with these efforts and is in fact being helpful.


“The US contribution was as follows: the removal of export barriers for Russian fertilizers, unblocking ports, [lifting restrictions on] banking transactions, etc. But some countries from Europe wanted to sabotage it,” he said, signaling that Turkey continues to work to make sure the grain deal is upheld.
The agreement to unblock grain exports via the Black Sea was signed by Moscow and Russia at UN-brokered talks in Istanbul in late July, and aims to maintain safe transit routes. The agreement is also supposed to allow Russia to deliver fertilizers and food products to global markets. Many experts and officials deem the agreement to be instrumental in alleviating global food security issues.

Wheat deliveries from Ukraine, a major producer, were disrupted after Russia launched its military operation in the neighboring state in late February. The sides blamed each other for causing the crisis.

Last week, Russian President Vladimir Putin accused the US of attempting to prolong the Ukraine conflict by “pumping the Kiev regime with weapons, including heavy weapons.”

Putin stated that the Ukrainians have been assigned the role of “cannon fodder” in Washington’s “anti-Russia project.” The president also said that Moscow launched its offensive in Ukraine to “ensure the security of Russia and its citizens, and defend the people of Donbass from genocide.”

US President Joe Biden said in June that NATO will support Ukraine “as long as it takes” to make sure Kiev is not defeated."""

test_real = """The Federal Reserve on Wednesday enacted its second consecutive
            0.75 percentage point interest rate increase as it seeks to
            tamp down runaway inflation without creating a recession.

            In taking the benchmark overnight borrowing rate up to a range of 2.25%-2.5%, the moves in June and July represent the most stringent consecutive action since the Fed began using the overnight funds rate as the principal tool of monetary policy in the early 1990s.

While the fed funds rate most directly impacts what banks charge each other for short-term loans, it feeds into a multitude of consumer products such as adjustable mortgages, auto loans and credit cards. The increase takes the funds rate to its highest level since December 2018.

Markets largely expected the move after Fed officials telegraphed the increase in a series of statements since the June meeting. Stocks hit their highs after Fed Chair Jerome Powell left the door open about its next move at the September meeting, saying it would depend on the data. Central bankers have emphasized the importance of bringing down inflation even if it means slowing the economy."""

test_real_2 = """Pakistani Prime Minister Shehbaz Sharif joined in
relief efforts over the weekend, dropping off supplies from a helicopter in
areas difficult to reach by boat or land, according to videos from his office.
"Visiting flood affected areas and meeting people. The magnitude of the calamity
is bigger than estimated," Sharif said in a tweet on Saturday. "Times demand
that we come together as one nation in support of our people facing this
calamity. Let us rise above our differences and stand by our people who need
us today." After meeting with ambassadors and diplomats in Islamabad on Friday,
he called for help from the international community. Residents gather beside a
road damaged by flood waters following heavy monsoon rains in Charsadda district
of Khyber Pakhtunkhwa on August 29, 2022. Residents gather beside a road damaged
by flood waters following heavy monsoon rains in Charsadda district of Khyber Pakhtunkhwa on August 29, 2022. On Monday, Peter Ophoff, the IFRC head delegate in Pakistan said the aid network had appealed for more than $25 million to provide urgent relief for an estimated 324,000 people in the country. "Looking at the incredible damage the floods have caused, it slowly becoming clear to us that relief efforts are going to take a very long time. It is going to be a long-waterlogged road ahead when the people of Pakistan began their journey back to what is remaining of their homes," Ophoff said. See volunteers use bedframe to rescue people from deadly floods 00:52 More than 3.1 million people had been displaced by the "sea-like" flood waters that have damaged more than half a million homes in multiple districts across the country, according to a statement Saturday from the International Federation of Red Cross and Red Crescent Societies (IFRC). Abrar ul Haq, chairman of the aid network in Pakistan, said Friday that water wasnt the only challenge for humanitarian workers in the region. "These torrential floods have severely restricted transportation and mobility. The threat of Covid-19 and damage to vehicles, infrastructure and connectivity are further making our emergency relief works almost impossible. Most of those affected are also immobile or marooned making us hard to reach them," he said. Displaced people prepare for breakfast in their tents at a makeshift camp after fleeing from their flood-hit homes following heavy monsoon rains in Charsadda district of Khyber Pakhtunkhwa on August 29, 2022. Displaced people prepare for breakfast in their tents at a makeshift camp after fleeing from their flood-hit homes following heavy monsoon rains in Charsadda district of Khyber
Pakhtunkhwa on August 29, 2022. Monster monsoon of the decade"""

news_notnews(test_real_2)

test_not_news = """I saw the best minds of my generation destroyed by madness, starving hysterical naked,
dragging themselves through the negro streets at dawn looking for an angry fix,
angelheaded hipsters burning for the ancient heavenly connection to the starry dynamo in the machinery of night,
who poverty and tatters and hollow-eyed and high sat up smoking in the supernatural darkness of cold-water flats floating across the tops of cities contemplating jazz,
who bared their brains to Heaven under the El and saw Mohammedan angels staggering on tenement roofs illuminated,
who passed through universities with radiant cool eyes hallucinating Arkansas and Blake-light tragedy among the scholars of war,
who were expelled from the academies for crazy & publishing obscene odes on the windows of the skull,
who cowered in unshaven rooms in underwear, burning their money in wastebaskets and listening to the Terror through the wall,
who got busted in their pubic beards returning through Laredo with a belt of marijuana for New York,
who ate fire in paint hotels or drank turpentine in Paradise Alley, death, or purgatoried their torsos night after night
with dreams, with drugs, with waking nightmares, alcohol and cock and endless balls,
incomparable blind streets of shuddering cloud and lightning in the mind leaping toward poles of Canada & Paterson, illuminating all the motionless world of Time between,
Peyote solidities of halls, backyard green tree cemetery dawns, wine drunkenness over the rooftops, storefront boroughs of teahead joyride neon blinking traffic light, sun and moon and tree vibrations in the roaring winter dusks of Brooklyn, ashcan rantings and kind king light of mind,
who chained themselves to subways for the endless ride from Battery to holy Bronx on benzedrine until the noise of wheels and children brought them down shuddering mouth-wracked and battered bleak of brain all drained of brilliance in the drear light of Zoo,
who sank all night in submarine light of Bickford’s floated out and sat through the stale beer afternoon in desolate Fugazzi’s, listening to the crack of doom on the hydrogen jukebox,
who talked continuously seventy hours from park to pad to bar to Bellevue to museum to the Brooklyn Bridge,
a lost battalion of platonic conversationalists jumping down the stoops off fire escapes off windowsills off Empire State out of the moon,
yacketayakking screaming vomiting whispering facts and memories and anecdotes and eyeball kicks and shocks of hospitals and jails and wars,
whole intellects disgorged in total recall for seven days and nights with brilliant eyes, meat for the Synagogue cast on the pavement,
who vanished into nowhere Zen New Jersey leaving a trail of ambiguous picture postcards of Atlantic City Hall"""
