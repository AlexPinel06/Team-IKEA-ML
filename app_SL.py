import streamlit as st
import joblib

# Charger le modèle, le vectoriseur et l'encodeur de labels
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Prédiction de la difficulté des phrases")

sentence = st.text_input("Entrez une phrase :")

if sentence:
    X_tfidf = vectorizer.transform([sentence])
    prediction = model.predict(X_tfidf)
    difficulty = label_encoder.inverse_transform(prediction)
    st.write(f"La difficulté prédite pour la phrase est : {difficulty[0]}")
