import streamlit as st
import joblib
import os

# Afficher le répertoire courant pour vérifier les chemins
st.write("Répertoire courant :", os.getcwd())
st.write("Fichiers dans le répertoire courant :", os.listdir('.'))

# Charger le modèle, le vectoriseur et l'encodeur de labels
try:
    model = joblib.load('model2/logistic_regression_model.pkl')
    st.write("Modèle chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du modèle : {e}")

try:
    vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
    st.write("Vectorizer chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du vectorizer : {e}")

try:
    label_encoder = joblib.load('model2/label_encoder.pkl')
    st.write("Label encoder chargé avec succès.")
except Exception as e:
    st.write(f"Erreur lors du chargement du label encoder : {e}")

st.title("Prédiction de la difficulté des phrases")

sentence = st.text_input("Entrez une phrase :")

if sentence:
    try:
        X_tfidf = vectorizer.transform([sentence])
        prediction = model.predict(X_tfidf)
        difficulty = label_encoder.inverse_transform(prediction)
        st.write(f"La difficulté prédite pour la phrase est : {difficulty[0]}")
    except Exception as e:
        st.write(f"Erreur lors de la prédiction : {e}")
