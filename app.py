import os
import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F
from transformers import CamembertForSequenceClassification, CamembertTokenizer

# Chemins vers les fichiers dans le dépôt GitHub
MODEL_PATH = 'model/trained_model.pt'
CLASSES_PATH = 'model/classes.npy'

@st.cache(allow_output_mutation=True)
def load_model():
    st.write(f"Chargement du modèle depuis {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        st.error(f"Le fichier modèle n'existe pas à l'emplacement {MODEL_PATH}")
        return None
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=10)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache
def load_classes():
    st.write(f"Chargement des classes depuis {CLASSES_PATH}")
    if not os.path.exists(CLASSES_PATH):
        st.error(f"Le fichier des classes n'existe pas à l'emplacement {CLASSES_PATH}")
        return None
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return classes

def predict_difficulty(model, tokenizer, text):
    # Transformation du texte en entrée du modèle
    st.write("Transformation du texte pour la prédiction")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted.item()

def main():
    st.title("Prédiction de la difficulté d'une phrase")

    model = load_model()
    if model is None:
        st.error("Le modèle n'a pas pu être chargé.")
        return
    
    classes = load_classes()
    if classes is None:
        st.error("Les classes n'ont pas pu être chargées.")
        return

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    
    text = st.text_input("Entrez une phrase :")

    if text:
        difficulty_index = predict_difficulty(model, tokenizer, text)
        difficulty = classes[difficulty_index]
        st.write(f"La difficulté de la phrase est : {difficulty}")

if __name__ == "__main__":
    main()
