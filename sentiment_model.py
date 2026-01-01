import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "review": [
        # affirmative ones
        "This movie was fantastic and very entertaining",
        "I loved the storyline and the acting",
        "Amazing visuals and great direction",
        "Absolutely wonderful experience",
        "A brilliant film with strong performances",

        # negative 
        "Terrible movie, waste of time",
        "The film was boring and poorly executed",
        "Worst movie I have ever seen",
        "Not good, very disappointing",
        "A complete mess with no proper storyline",

        # sarcastic
        "The idea was good but the execution was terrible",
        "I wanted to like this movie but it just did not work",
        "Some great moments but overall a disappointing film",
        "The movie tried too hard and ended up being boring",
        "Fantabulous visuals but the story was bogus",
        "Not bad, but definitely not as good as the hype"
    ],
    "sentiment": [
        1, 1, 1, 1, 1,     
        0, 0, 0, 0, 0,     
        0, 0, 0, 0, 0, 0  
    ]
}

df= pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']


model = LogisticRegression()
model.fit(X, y)

def predict_sentiment(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return "Positive :D " if prediction[0] == 1 else "Negative :( "
