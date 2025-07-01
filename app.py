import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load and prepare data (as in your notebook)
df = pd.read_csv("news.csv")
x = df['text']
y = df.iloc[:,-1]

# Vectorizer and model training (same as notebook)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf = tfidf_vectorizer.fit_transform(x)
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf, y)

st.title('Fake News Detector')
st.write('Enter news text below to check if it is FAKE or REAL:')

user_text = st.text_area('News text')
if user_text:
    user_tfidf = tfidf_vectorizer.transform([user_text])
    user_pred = pac.predict(user_tfidf)[0]
    st.write(f'Prediction: **{user_pred}**')
