import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import re
import os

# Fonction pour encoder les textes
def encode_text(data, tokenizer):
    sentences = data['sentence'].tolist()
    encodings = tokenizer(sentences, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    encodings['text_length'] = torch.tensor(data['text_length'].tolist())
    encodings['punctuation_count'] = torch.tensor(data['punctuation_count'].tolist())
    return encodings

# Chemins vers les fichiers de modèle et les classes
model_path = 'model/trained_model.pt'
classes_path = 'model/classes.npy'

# Vérification de l'existence des fichiers nécessaires
if not os.path.exists(model_path):
    st.error(f"Le fichier de modèle {model_path} n'existe pas.")
    st.stop()

if not os.path.exists(classes_path):
    st.error(f"Le fichier des classes {classes_path} n'existe pas.")
    st.stop()

# Chargement du tokenizer et du modèle
try:
    tokenizer_camembert = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)  # Ajustez le nombre de labels si nécessaire
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except ImportError as e:
    st.error(f"Erreur d'importation : {e}")
    st.stop()

# Chargement de l'encodeur de labels
try:
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(classes_path, allow_pickle=True)
except Exception as e:
    st.error(f"Erreur lors du chargement des classes : {e}")
    st.stop()

# Titre de l'application
st.title('Prédiction du Niveau de Difficulté des Phrases en Français')

# Champ de texte pour entrer une phrase
sentence = st.text_input('Entrez une phrase en français:')

# Bouton pour prédire la difficulté
if st.button('Prédire la Difficulté'):
    if sentence:
        # Préparer les données
        df = pd.DataFrame({'sentence': [sentence]})
        df['text_length'] = df['sentence'].apply(len)
        df['punctuation_count'] = df['sentence'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
        encodings = encode_text(df, tokenizer_camembert)

        # Faire la prédiction
        with torch.no_grad():
            outputs = model(**encodings)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]

        # Afficher le résultat
        st.write(f'Le niveau de difficulté de la phrase est: {predicted_label}')
    else:
        st.write('Veuillez entrer une phrase.')
