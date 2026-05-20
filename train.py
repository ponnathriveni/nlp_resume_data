

import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


df = pd.read_csv("resume.csv")
print(df.columns)


df.columns = df.columns.str.strip().str.lower()


df = df.rename(columns={
    "text": "content",
    "label": "annotation"
})


df = df[['content', 'annotation']].dropna()

print(df.head())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    
    text = re.sub(r'http\S+|www\S+', '', text)

    
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    
    words = text.split()

    
    words = [word for word in words if word not in stop_words]

    
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


df['content'] = df['content'].apply(clean_text)


X = df['content']
y = df['annotation']


tfidf = TfidfVectorizer(max_features=5000)

X_vectorized = tfidf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


model = MultinomialNB()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)


pickle.dump(model, open("model.pkl", "wb"))
#``````pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
print("Model Saved Successfully")