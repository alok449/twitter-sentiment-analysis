import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#class TweetText(BaseModel):
#    tweet: str

class TweetModel:
    def __init__(self):
        self.df = pd.read_csv('twitter_new.csv', header=None, encoding='latin-1')
        self.df = self.clean_prepare_dataset()
        self.model_fname_ = 'twitter_sentiment_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def clean_prepare_dataset(self):
        self.df = self.df.loc[:, [0,5]]
        self.df = self.df.rename(columns={0:"sentiment", 5:"text"})
        df_text = self.df['text']
        df_sentiment = self.df['sentiment']
        self.df.insert(0, "text_new", df_text)
        self.df.insert(1, "sentiment_new", df_sentiment)
        self.df.drop(['sentiment','text'], axis=1, inplace=True)
        self.df = self.df.rename(columns={"text_new":"text", "sentiment_new":"sentiment"})
        return self.df

    def filteredWord(self):
        corpus = []
        for i in range(0, len(self.df)):
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",self.df['text'][i]).split())
            tweet = tweet.lower()
            tweet = tweet.split()
            ps = PorterStemmer()
            tweet = [ps.stem(word) for word in tweet if word not in set(stopwords.words('english'))]
            tweet = ' '.join(tweet)
            corpus.append(tweet)
        return corpus

    def _train_model(self):
        #cv = CountVectorizer(max_features = 1500)
        #X = cv.fit_transform(self.filteredWord()).toarray()
        X = self.cvModel().transform(self.filteredWord()).toarray()
        #print('shape of trainin model')
        #print(X.shape)
        y = self.df['sentiment']
        svc = SVC(kernel='rbf')
        model = svc.fit(X,y)
        return model

    def cvModel(self):
        cv = CountVectorizer(max_features = 1500)
        cvModel = cv.fit(self.filteredWord())
        return cvModel

    def predict_sentiment(self, tweet):
        #tweet = tweet.split()
        tweet_in = []
        #tweet_in = ' '.join(tweet)
        tweet_in.append(tweet)
        print(tweet_in)
        tweet_in = self.cvModel().transform(tweet_in).toarray()
        #print(tweet_in.shape)
        #tweet_in = tweet_in.reshape(len(tweet_in),1500)
        print(tweet_in.shape)
        prediction = self.model.predict(tweet_in)
        print(prediction)
        return prediction[0]

