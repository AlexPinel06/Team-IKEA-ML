import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the model, vectorizer, and label encoder
model = joblib.load('model2/logistic_regression_model.pkl')
vectorizer = joblib.load('model2/tfidf_vectorizer.pkl')
label_encoder = joblib.load('model2/label_encoder.pkl')

# Function to reset the input
def reset_input():
    if "random_questions" in st.session_state:
        del st.session_state["random_questions"]
    if "random_levels" in st.session_state:
        del st.session_state["random_levels"]
    st.session_state["user_written_sentences"] = [""] * 6
    st.experimental_rerun()

# Function to calculate the score
def calculate_score(user_answers, correct_answers):
    return sum(1 for user, correct in zip(user_answers, correct_answers) if user == correct)

# Function to update scores
def update_scores(new_score):
    if "scores" not in st.session_state:
        st.session_state["scores"] = []
    st.session_state["scores"].append(new_score)
    if len(st.session_state["scores"]) > 3:
        st.session_state["scores"] = st.session_state["scores"][-3:]

# Determine the level based on the score
def get_level(score):
    if 1 <= score <= 2:
        return 'A1'
    elif 3 <= score <= 4:
        return 'A2'
    elif 5 <= score <= 6:
        return 'B1'
    elif 7 <= score <= 8:
        return 'B2'
    elif 9 <= score <= 10:
        return 'C1'
    else:
        return 'C2'

# Initialize session state
if "scores" not in st.session_state:
    st.session_state["scores"] = []

if "random_questions" not in st.session_state:
    st.session_state["random_questions"] = []

if "random_levels" not in st.session_state:
    st.session_state["random_levels"] = []

if "user_written_sentences" not in st.session_state:
    st.session_state["user_written_sentences"] = [""] * 6

# Questions and correct answers
given_sentences = [
    "Je suis étudiant.",
    "Il fait beau aujourd'hui.",
    "La théorie de la relativité est complexe.",
    "Les ordinateurs quantiques sont l'avenir.",
    "Il a couru un marathon hier.",
    "Elle aime les animaux.",
    "Ils ont déménagé en France.",
    "Nous allons à la plage.",
    "Je préfère le café au thé.",
    "Le réchauffement climatique est une menace."
]
correct_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'A1', 'A2', 'B1', 'B2', 'C1']
difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# Example sentences for each level
example_sentences = {
    'A1': "Je m'appelle Marie.",
    'A2': "J'aime le chocolat.",
    'B1': "Je vais à l'université en bus.",
    'B2': "La technologie moderne révolutionne notre quotidien.",
    'C1': "L'innovation est essentielle pour la compétitivité des entreprises.",
    'C2': "La théorie des cordes est une perspective fascinante de la physique moderne."
}

# YouTube videos for each level
youtube_videos = {
    'A1': "https://www.youtube.com/watch?v=4w6REQGQ89E",
    'A2': "https://www.youtube.com/watch?v=dh7O5BV47lU",
    'B1': "https://www.youtube.com/watch?v=CwXjunivNbk",
    'B2': "https://www.youtube.com/watch?v=8J76QTENZv4",
    'C1': "https://www.youtube.com/shorts/4jZE6y-uGJQ",
    'C2': "https://www.youtube.com/watch?v=rRgTKrp1hVc"
}

# Randomize questions and levels if not already randomized
if not st.session_state["random_questions"]:
    combined = list(zip(given_sentences, correct_levels))
    random.shuffle(combined)
    st.session_state["random_questions"] = combined[:6]  # Select 6 random questions

if not st.session_state["random_levels"]:
    st.session_state["random_levels"] = random.sample(difficulty_levels, 6)

random_questions = st.session_state["random_questions"]
given_sentences = [q[0] for q in random_questions]
correct_levels = [q[1] for q in random_questions]
random_levels = st.session_state["random_levels"]

# User answers
user_answers = []
correct_answers = correct_levels + random_levels

# First 6 questions: Select difficulty level for given sentences
st.subheader("Select the difficulty level for the given sentences:")
for i, sentence in enumerate(given_sentences):
    user_answers.append(st.selectbox(f"Sentence {i+1}: {sentence}", options=difficulty_levels))

# Last 6 questions: Write sentences for given difficulty levels
st.subheader("Write a sentence for each given difficulty level:")
user_written_sentences = []
for i, level in enumerate(random_levels):
    user_sentence = st.text_input(f"Write a sentence for level {level}:", value=st.session_state["user_written_sentences"][i], key=f"user_sentence_{i}")
    user_written_sentences.append(user_sentence)
    st.session_state["user_written_sentences"][i] = user_sentence
    if user_sentence:
        X_tfidf = vectorizer.transform([user_sentence])
        prediction = model.predict(X_tfidf)
        predicted_level = label_encoder.inverse_transform(prediction)[0]
        user_answers.append(predicted_level)

# Calculate score
if st.button("Submit"):
    score = calculate_score(user_answers, correct_answers)
    update_scores(score)
    st.write(f"Your score is: {score}/12")
    
    level = get_level(score)
    st.write(f"Your French level is: {level}")
    st.markdown("""
    Your French level is determined by the number of points you score out of 12.
    - 1-2 points: A1
    - 3-4 points: A2
    - 5-6 points: B1
    - 7-8 points: B2
    - 9-10 points: C1
    - 11-12 points: C2
    """)

    st.subheader("Your Answers")
    for i, (user_answer, correct_answer) in enumerate(zip(user_answers[:6], correct_answers[:6])):
        if user_answer == correct_answer:
            st.write(f"Question {i+1}: Correct! {user_answer} ✅")
        else:
            st.write(f"Question {i+1}: Incorrect! {user_answer} ❌ (Correct: {correct_answer})")

    st.subheader("Analysis of Your Written Sentences")
    for i, (user_sentence, level) in enumerate(zip(user_written_sentences, random_levels)):
        if user_sentence:
            X_tfidf = vectorizer.transform([user_sentence])
            prediction = model.predict(X_tfidf)
            predicted_level = label_encoder.inverse_transform(prediction)[0]
            example_sentence = example_sentences[level]
            if predicted_level == level:
                st.write(f"Sentence {i+7}: Correct! Predicted Level: {predicted_level} ✅")
            else:
                st.write(f"Sentence {i+7}: Incorrect! Predicted Level: {predicted_level} ❌ (Expected Level: {level})")
            st.write(f"Example sentence for level {level}: {example_sentence}")

    
    # Display the French level
    level = get_level(score)
    st.subheader("Your French Level")
    st.write(f"Your French level based on your score is: **{level}**")

    # Estimated data for illustration
    difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    counts = [30, 25, 20, 15, 7, 3]  # Estimated percentage for each difficulty level

    # Display a pie chart of difficulty levels
    st.subheader("Estimated Distribution of French Learners by Competency Level")
    fig, ax = plt.subplots(facecolor='#0e1117')  # Set the background color of the figure
    wedges, texts, autotexts = ax.pie(counts, labels=difficulty_levels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Customize the appearance
    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('white')
    fig.patch.set_facecolor('#0e1117')  # Set the background color of the plot area

    st.pyplot(fig)

    st.subheader("Your Ranking")
    better_than_percentage = sum(counts[:difficulty_levels.index(level)])
    top_percent = 100 - better_than_percentage
    st.write(f"You belong to the top **{top_percent}%** of all users with this difficulty level.")

    # Fun facts for each level
    fun_facts = {
        'A1': "Fun fact: Even Einstein had to start somewhere!",
        'A2': "Fun fact: You're now better than most tourists!",
        'B1': "Fun fact: You're officially conversational!",
        'B2': "Fun fact: You can enjoy French movies without subtitles!",
        'C1': "Fun fact: Your French is better than most expats!",
        'C2': "Fun fact: You're at the mastery level, like a true Parisian!"
    }
    st.write(fun_facts[level])

    # Add source information
    st.markdown("""
    **Sources:**
    - Portal (CECR)
    - French Together – Learn French
    - Service-Public
    - Kwiziq French
    - FluentU
    - **Note:** These percentages are based on general estimates and information available on the distribution of language proficiency levels according to the CEFR in various educational and linguistic sources.
    """)




    
    # Display the corresponding YouTube video
    st.subheader("Watch a video for your level")
    st.markdown("""
    For a comparison, here's an English speaker at your level (of course, these are jokes to be taken in the second degree. They are memes very famous on the internet):
    """)
    st.video(youtube_videos[level])

# Button to reset the input
if st.button("Reset"):
    reset_input()

# Display the last three scores
st.sidebar.title("Your Last 3 Scores")
for score in st.session_state["scores"]:
    st.sidebar.write(f"Score: {score}/12")

# Add a sidebar with additional information
st.sidebar.title("About")
st.sidebar.info("""
This app uses a pre-trained model to predict the difficulty level of sentences.
The model was trained on a dataset of sentences labeled with difficulty levels.
""")
st.sidebar.title("Instructions")
st.sidebar.info("""
1. Select the difficulty level for each of the first 6 sentences.
2. Write a sentence for each of the last 6 difficulty levels.
3. Submit your answers to see your score (and get more fun facts :)).
""")

# Add extra information in the sidebar
st.sidebar.title("Fun Facts")
st.sidebar.info("""
- A1 is the beginner level, while C2 is the mastery level.
- Difficulty prediction can help in language learning by tailoring content to your level.
- Natural Language Processing (NLP) techniques are used to analyze and understand human language.
- The model is a state-of-the-art model for sequence classification tasks.
""")

# Footer
st.markdown("""
---
Developed by Igor Dallemagne and Alex Pinel.
""")
