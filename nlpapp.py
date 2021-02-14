import uvicorn
from fastapi import FastAPI
from nlpmodel import  TweetModel

app = FastAPI()
model = TweetModel()

@app.post('/predictsentiment')
def predict_sentiment(tweet):
    data = tweet
    #print(data)
    prediction = model.predict_sentiment(
        data
    )
    print(prediction)
    return {'prediction': str(prediction)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port='8000')