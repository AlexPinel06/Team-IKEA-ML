import streamlit as st
import torch
import numpy as np
import torch.nn.functional as F

# Chemins vers les fichiers dans le dépôt GitHub
MODEL_PATH = 'model/trained_model.pt'
CLASSES_PATH = 'model/classes.npy'

@st.cache(allow_output_mutation=True)
def load_model():
    # Charger le modèle depuis le fichier local
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache
def load_classes():
    # Charger les classes depuis le fichier local
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return classes

def predict_difficulty(model, text):
    # Transformation du texte en entrée du modèle
    # Cette partie doit être adaptée à votre modèle
    # Exemple : encoder le texte avec une méthode simple (à adapter selon votre modèle)
    # Note : Cela suppose que votre modèle accepte des chaînes de caractères directement, ce qui est rare.
    # Vous devrez probablement effectuer un prétraitement spécifique à votre modèle ici.
    
    # Ici, on suppose que le modèle accepte une liste de mots sous forme de tenseur
    inputs = torch.tensor([len(text.split())], dtype=torch.float).unsqueeze(0)  # Exemple simple basé sur la longueur du texte
    with torch.no_grad():
        outputs = model(inputs)
    probabilities = F.softmax(outputs, dim=1)
    _, predicted = torch.max(probabilities, 1)
    return predicted.item()

def main():
    st.title("Prédiction de la difficulté d'une phrase")

    model = load_model()
    classes = load_classes()

    text = st.text_input("Entrez une phrase :")

    if text:
        difficulty_index = predict_difficulty(model, text)
        difficulty = classes[difficulty_index]
        st.write(f"La difficulté de la phrase est : {difficulty}")

if __name__ == "__main__":
    main()
