import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import re
import os
import gdown

# Fonction pour télécharger le modèle depuis Google Drive
def download_file_from_google_drive(file_id, dest):
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', dest, quiet=False)

# IDs des fichiers sur Google Drive
model_file_id = 'https://drive.google.com/file/d/1vQkZ_OXSpjNyE1J9Hy8wX5xj0x4fpAx0/view?usp=share_link'
classes_file_id = 'https://drive.google.com/file/d/11Hx_s74IMEc1DdQ6YaESI9JJdqhY81No/view?usp=share_link'

# Chemins vers les fichiers de modèle et les classes
model_path = 'model/trained_model.pt'
classes_path = 'model/classes.npy'

# Créer le répertoire du modèle s'il n'existe pas
if not os.path.exists('model'):
    os.makedirs('model')

# Télécharger les fichiers si nécessaire
if not os.path.exists(model_path):
    download_file_from_google_drive(model_file_id, model_path)

if not os.path.exists(classes_path):
    download_file_from_google_drive(classes_file_id, classes_path)

# Fonction pour encoder les textes
def encode_text(data, tokenizer):
    sentences = data['sentence'].tolist()
    encodings = tokenizer(sentences, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    encodings['text_length'] = torch.tensor(data['text_length'].tolist())
    encodings['punctuation_count'] = torch.tensor(data['punctuation_count'].tolist())
    return encodings

# Chargement du tokenizer et du modèle
try:
    tokenizer_camembert = CamembertTokenizer.from_pretrained('camembert-base')
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=3)
    
    # Vérifier si le fichier du modèle est valide
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.eval()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
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
        try:
            with torch.no_grad():
                outputs = model(**encodings)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]

            # Afficher le résultat
            st.write(f'Le niveau de difficulté de la phrase est: {predicted_label}')
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
    else:
        st.write('Veuillez entrer une phrase.')
