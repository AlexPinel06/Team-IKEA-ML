# EPFL_IKEA: AI-Powered French Text Difficulty Prediction
## Link for the app: https://team-ikea-ml-73y8jzua6jrjzfpyemvpjr.streamlit.app
## Link for the video : https://youtu.be/gVDh_myY_Kw

## Introduction 

In the burgeoning field of language learning technology, accurately gauging text complexity can significantly enhance learning outcomes. This GitHub repository, EPFL_IKEA, is the culmination of efforts by us to create an AI model that predicts the difficulty levels of French texts. Our journey, rooted in our own experiences as native French speakers and learners of English, has led us from competing in Kaggle to developing a robust tool that aids learners by matching them with appropriately challenging texts.

## Project Background

The ability to correctly match text difficulty with a learner's proficiency level is pivotal in language education. It ensures that learners are neither overwhelmed by complexity beyond their comprehension nor under-challenged by overly simplistic texts. Recognizing this, we embarked on creating a machine learning model capable of classifying French texts across the Common European Framework of Reference for Languages (CEFR) levels from A1 (beginner) to C2 (advanced).

## Data Resources

The foundation of our project is a dataset consisting of 4,800 French sentences, each tagged with a CEFR difficulty level. We utilized the following datasets throughout our project:

- **Training Data:** Contains the sentences used to train our models. [Access the Training Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/training_data.csv)
- **Unlabelled Test Data:** Used for predicting sentence difficulty as part of the competition based on accuracy score. [Access the Test Dataset](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/unlabelled_test_data.csv)
- **Sample Submission:** Demonstrates the submission format for the competition. [Download the Sample Submission File](https://raw.githubusercontent.com/pinoulex/Team-IKEA-ML/main/data/sample_submission.csv)

## Model Development and Performance

### Initial Machine Learning Approaches

Our journey began by testing traditional machine learning methods:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

### Advanced Modeling with BERT

Progressing to more sophisticated models, we incorporated BERT, specifically its French variant Camembert, due to its advanced capabilities in understanding contextual nuances in language.

### Performance Metrics Overview

We assess our models using key metrics to ensure precision, recall, F1-score, and accuracy. Here's how each model performed:


|                  | Logistic Regression | KNN     | Decision Tree | Random Forest | Camembert |
|------------------|---------------------|---------|---------------|---------------|-----------|
| Precision        |         0.44        |   0.29  |      0.31     |      0.39     |   0.82    |
| Recall           |         0.45        |   0.20  |      0.31     |      0.37     |   0.82    |
| F1-score         |         0.44        |   0.11  |      0.31     |      0.35     |   0.82    |
| Accuracy         |         0.45        |   0.20  |      0.31     |      0.37     |   0.82    |


Understanding the differences in these metrics is crucial for comprehending model performance. **Precision** quantifies the accuracy of positive <u>predictions</u>, highlighting the proportion of true positives among all positive predictions. **Recall** assesses the model's ability to identify all relevant instances, focusing on the proportion of actual positives correctly identified. **F1-score** balances precision and recall, useful in situations with uneven class distributions. **Accuracy** reflects the overall correctness of the model across all predictions, relevant when the classes are symmetric in size and the costs of false positives and false negatives are similar.

In our case, the respective subsections are more or less even, suggesting balanced model performance in terms of handling different classes and types of errors. In subsequent sections, we will explore each model's advantages and limitations in detail.

### Logistic Regression

We selected Logistic Regression as our initial method to explore predicting sentence difficulties using machine learning. This model stands out for its computational efficiency, facilitating swift training and prediction processes. Logistic Regression operates by first converting sentences into numerical form—specifically, by creating features based on the characteristics of the sentences, such as word frequency, sentence length, and syntax complexity. The model then applies these numerical features to a logistic function, which estimates the probability that a sentence belongs to a particular difficulty category. A significant strength of Logistic Regression is its interpretability; it provides clear insights by showing how each feature’s weight (regression coefficient) influences the predicted difficulty level, allowing for easier adjustments and understanding of the model's decisions.

We split the training data using an 80/20 ratio, with 80% used for training the model and the remaining 20% for testing. The results, as shown in the table above, reflect this setup. The regression model slightly outperforms other traditional models, but it only moderately succeeds in accurately determining sentence difficulties, achieving Precision and Recall rates of 44% and 45%, respectively. This means the model correctly predicts sentence difficulty 44% of the time and accurately identifies 45% of all actual difficult sentences.

To delve deeper into where the model may be going wrong, we analyzed the confusion matrix, which highlights the model's tendency to misclassify sentences, particularly those in the middle difficulty range (B2 to C1). These sentences likely have features that overlap significantly with adjacent difficulty levels, unlike sentences at the extremes, which tend to have more distinct characteristics.

<img width="500" alt="Capture d’écran 2024-05-19 à 17 41 00" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/51a483f3-839e-44d2-847f-2022a63bdc01">

A case in point is the misclassified sentence "Un service précieux qui a sans doute permis de sauver des vies," incorrectly labeled as C1 instead of its actual difficulty of A2. Analysis of its vectorization and contributing predictors shows that the most influential features leading to the incorrect C1 classification were common determiners and prepositions like 'des', 'sa', 'les', 'de', and 'du'. These frequently used words are typically neutral concerning difficulty and indicate that the model might struggle with capturing the nuances of technical vocabulary, grammar, or punctuation. Notably, attempts to improve model performance by removing these "stop words" resulted in poorer outcomes with an accuracy of 42%, suggesting that common words still hold significant predictive value. Nevertheless, the model does recognize that longer sentences generally pose more difficulty, as evidenced by correctly classifying the complex sentence: "En attribuant une valeur de flux potentiel aux liens du graphe, la segmentation fondée sur le critère de modularité aboutit à un découpage considéré comme optimal du point de vue de la dynamique démographique interne à chaque compartiment."

These insights highlight the need for further refinement to better distinguish between linguistic complexity and commonplace vocabulary in our model. Below is the code we used for this analysis:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

df = df_training.copy()

X = df['sentence']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

metrics = precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
print(f"Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}, Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

### KNN

Following our use of Logistic Regression, we explored another approach using the k-Nearest Neighbors (kNN) classification algorithm to predict sentence difficulties. This method categorizes sentences by assigning them to predefined difficulty levels based on their features. Specifically, kNN operates by identifying a specified number of training examples that are closest to a new sentence in terms of feature similarity—such as sentence length and complexity. It then predicts the difficulty level of the new sentence based on the most common categories among these nearest neighbors. This method relies heavily on the assumption that similar sentences share similar difficulty levels, making it particularly effective when clear patterns of similarity exist within the data.

Using the 80/20 split technique again, we obtained the results presented in the initial table. The outcome suggests poor prediction performance, with a Precision of 29%. However, the identification capabilities are even lower, with a Recall of 20%, resulting in more false positives than false negatives.

Examining the confusion matrix provides insights into why the model fails to properly assess difficulty. It reveals that the vast majority of predictions are for A1, resulting in many false positives. The model’s performance might be hindered by the high dimensionality of text data, which results from the large number of features derived from each word. This complexity makes it more challenging to identify clear patterns and accurately find nearest neighbors. This stresses the poor scalability of the kNN model.

<img width="500" alt="Capture d’écran 2024-05-18 à 19 12 59" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/ca1916f7-1333-440d-a50d-28909fa3c0e5">

To delve deeper, we analyzed the most frequent words in misclassified sentences (stop words excluded while counting), including words like 'plus', 'comme', 'cette', and 'si', even after excluding stop words into the model—which, once more, resulted in a poorer outcome. This shows that our model is overly influenced by common French adverb, adjective, and conjunctions, leading to clustering around the A1 prediction. Indeed, the concept of the "nearest" neighbor becomes less meaningful. Distances between points (words/ sentences) can become uniformly large or not sufficiently distinct to discern close from distant neighbors effectively. For instance, the sentence B2 was wrongly classified as A2, "Si cette dernière innovation voyait le jour, une de ses applications serait tout aussi impressionnante et inquiétante." contains occurrences of 'cette', and 'si'.

These results demonstrate that kNN is not well-suited for handling the complexities of written text. Here is the code we used for our analysis:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df_KNN = df_training.copy()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_KNN['sentence'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_KNN['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

metrics = precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}, Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}")
```

### Decision Tree

We further expanded our exploration of machine learning models for predicting sentence difficulties by implementing a Decision Tree algorithm. This classification method predicts the difficulty level of sentences by constructing a model that learns simple decision rules from the data features. In our application, the model is visualized as a tree structure: each internal node represents a "test" related to an attribute of the sentence, such as the presence of specific words or the complexity of the sentence structure. Each branch stemming from these nodes denotes the outcome of the test, and each leaf node corresponds to a class label, which in this case is the predicted difficulty level. This hierarchical approach allows the Decision Tree to make clear and logical decisions based on the hierarchical significance of sentence features, making it both interpretable and effective for tasks with well-defined rules and distinctions between categories.

Performing the 80/20 split technique again, we obtained results as shown in the table above, which suggest an even distribution among the classes and types of errors. However, the model still produces poor predictive results, with an accuracy of 31%.

Upon examining the confusion matrix, we observed that the model struggles to discern subtle difficulty distinctions, resulting in a significant number of false positives and negatives. This suggests an overfitting issue where the model captures noise instead of actual difficulty boundaries.   

<img width="500" alt="Capture d’écran 2024-05-19 à 10 31 28" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/4c4e5dee-f5c1-4925-a2ea-53558e75a6c2">

To analyze the erroneous predictions more closely, we examined the top 10 features that most significantly influence decision-making. Notably, these features include common French words such as 'de', 'la', 'les', and 'et'. Once again, excluding these stop words proved detrimental to the results, underscoring that our model is heavily influenced by common French articles, prepositions, and conjunctions. This leads to misinterpretations of true language difficulties because the distinctions are not well-defined, due to the algorithm's struggle with noisy labels and irrelevant features such as articles and propositions. Furthermore, because the model tends to overestimate the importance of common words, small variations in the data can result in the generation of a completely different decision tree.

These issues highlight the inherent limitations of such classification algorithms in addressing our predictive tasks. The code utilized for our analysis is provided below:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load and prepare the dataset
df_DT = df_training.copy()  # Assuming df_training is predefined
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_DT['sentence'])  # Convert text to word frequency vectors
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_DT['difficulty'])  # Encode labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict and calculate metrics
y_pred = dt_classifier.predict(X_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print results
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")
```
### Random Forest

Continuing with our exploration of machine learning techniques for predicting sentence difficulties, we implemented the Random Forest algorithm. This advanced classification method extends the concept of decision trees by creating an ensemble of decision trees trained on different subsets of the data and using different subsets of features at each split within those trees. Each tree in the forest makes its own prediction, and the majority vote across all trees determines the final output. This approach not only enhances the predictive accuracy by reducing the risk of overfitting associated with single decision trees but also maintains good interpretability, as it allows for an analysis of feature importance across multiple trees. Random Forest is particularly robust and effective in dealing with both large datasets and datasets with high dimensionality of features, making it well-suited for complex classification tasks like predicting the difficulty of sentences.

The 80/20 split yielded results, as shown in the introductory table, with a precision of 39% suggesting slightly better predictive capabilities compared to a recall of 37% for identification. We can see that the Random Forest, as expected, performs better than the other classification model, indicating better suitability to high dimensionality and overfitting. 

When examining the confusion matrix, similar to other classification methods, we noted that the algorithm predominantly predicts the A1 difficulty, leading to a higher number of false positives. These results suggest misclassification due to improper feature selection.

<img width="500" alt="Capture d’écran 2024-05-19 à 17 23 56" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/689f454c-6e9e-4972-86c2-de19aa5f603b">

To further investigate the errors, we analyzed the top 10 features influencing decision-making. Common articles and prepositions ('de', 'les', 'la', etc.) were identified, indicating that even the more robust model still tends to overestimate common words in predicting difficulty. Although the model shows improvement, it still exhibits inherent weaknesses typical of classification models. This has led us to consider transitioning to another type of algorithm for our prediction tasks.

You can find the code for our analysis detailed below:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df_RF = df_training.copy()  # Load your data

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_RF['sentence'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_RF['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {metrics[0]:.2f}, Recall: {metrics[1]:.2f}, F1-Score: {metrics[2]:.2f}, Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```
### Camembert

Based on the insights gained from previous results, we shifted our focus away from traditional machine learning models. Our exploration led us to the BERT model, discovered through Che-Jui Huang's 2022 Medium article, "[NLP] Project Review: Text Difficulty Classification." Unlike most linguistic models that process text unidirectionally, BERT, developed by Google, adopts a bidirectional approach. This means it reads and interprets text from both left to right and right to left simultaneously, allowing for a comprehensive understanding of context. For our specific case, we utilized Camembert, the French pre-trained model, to better handle the nuances of the French language.

In practical application, BERT evaluates the context surrounding each word by considering the words that precede and follow it. For example, in the French sentence "Je pars faire de la voile," BERT analyzes both "faire" and "la" when assessing the word "de." This deep contextual understanding enables BERT to accurately interpret grammatical roles and nuanced language use, such as determiners, more effectively than previous models. Initially, deploying BERT on our training dataset achieved an accuracy of 56%.

Unsatisfied with these results, we implemented a three-step enhancement strategy, leading to the improved outcomes reported in the initial table of 82%. These steps included data augmentation, hyperparameter optimization, and full-dataset training, ultimately yielding a final competition score of 0.611 on external data.

**Data Augmentation:**
We applied a two-pronged approach to data augmentation:

Synonym Replacement: Utilizing the WordNet database, we developed a function to find French synonyms, thereby expanding our dataset with alternative but semantically similar words. When no synonyms were available, the original word was retained.
Masking and Prediction: Using BERT’s predictive capabilities, we randomly masked words in sentences and allowed the model to infer suitable replacements based on the contextual clues from surrounding words. This not only diversified the sentence structures but also maintained the integrity of their original difficulty levels.
By doubling the dataset size through these methods, we enhanced its robustness and variability without sacrificing quality, thereby avoiding overfitting and maintaining realistic data variations.

**Hyperparameter Optimization:**
Hyperparameters are parameters whose values are set before the learning process begins and are not updated during training such as number of epochs, batch size, learning rate, etc. Using Optuna, we fine-tuned various model parameters identifying the optimal configuration that maximized performance on our specific tasks and dataset.

**Full-Dataset Training:**
Before entering the competition, we re-trained our model on the entire dataset, incorporating the 20% previously reserved for testing. This comprehensive exposure to all available data ensured that our model was well-prepared and had encountered the broadest possible spectrum of scenarios before facing the competition’s external dataset.

In summary, through careful data augmentation, meticulous hyperparameter tuning, and comprehensive training on the entire dataset, we significantly enhanced our model's performance, demonstrating robustness and adaptability in both controlled tests and competitive environments. These efforts improved the model's accuracy on an 80/20 data split to 82%, compared with a competition accuracy of 61%. This disparity suggests that while our data augmentation effectively increased the model's performance in a controlled setting, it may not have fully captured the broader nuances of the language required for the varied challenges presented in the competition.

When analyzing the confusion matrix, we can observe strong diagonal values, indicating good performance in correctly predicting each category. However, it is important to bear in mind the potential bias introduced by our data augmentation methods, as previously discussed.

<img width="500" alt="Capture d’écran 2024-05-23 à 23 01 39" src="https://github.com/igordall/EPFL_IKEA/assets/153678341/e01192a4-1c9e-4f26-919a-74210e3e2a5c">

Find below the code for Data Augmentation and Preparation: 
```python
def synonyms(word, lang='fra'):
    synsets = wordnet.synsets(word, lang=lang)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemma_names('fra'):
            synonyms.add(lemma.replace('_', ' '))
    return list(synonyms)

def replace_with_synonyms(sentence, lang='fra'):
    words = sentence.split()
    new_words = []
    for word in words:
        syns = synonyms(word, lang)
        if syns:
            new_word = random.choice(syns)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def mask_and_predict(sentence):
    words = sentence.split()
    masked_index = random.randint(0, len(words) - 1)
    words[masked_index] = '[MASK]'
    masked_sentence = ' '.join(words)
    tokenized_text = tokenizer_bert(masked_sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model_bert(**tokenized_text)
        predictions = outputs.logits
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer_bert.decode([predicted_index])
    words[masked_index] = predicted_token
    return ' '.join(words)

def encode_features(df):
    label_encoder = LabelEncoder()
    df['difficulty_encoded'] = label_encoder.fit_transform(df['difficulty'])
    df['text_length'] = df['sentence'].apply(len)
    df['punctuation_count'] = df['sentence'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
    return df, label_encoder

def encode_text(data, tokenizer):
    sentences = data['sentence'].tolist()
    encodings = tokenizer(sentences, padding='max_length', truncation=True, max_length=128)
    encodings['text_length'] = torch.tensor(data['text_length'].tolist())
    encodings['punctuation_count'] = torch.tensor(data['punctuation_count'].tolist())
    return encodings
df_training = augment_dataframe(df_training)
df_training, label_encoder = encode_features(df_training)
train_df, val_df = train_test_split(df_training, test_size=0.2, random_state=42)
train_encodings = encode_text(train_df, tokenizer_camembert)
val_encodings = encode_text(val_df, tokenizer_camembert)
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_df['difficulty_encoded'].tolist())
val_dataset = TextDataset(val_encodings, val_df['difficulty_encoded'].tolist())
```

Find below the code for the hyperparameters optimization: 
```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def objective(trial):
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=len(label_encoder.classes_))
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=6,
        per_device_train_batch_size=trial.suggest_int('per_device_train_batch_size', 4, 16),
        per_device_eval_batch_size=16,
        warmup_steps=trial.suggest_int('warmup_steps', 0, 500),
        weight_decay=trial.suggest_float('weight_decay', 0.0, 0.3),
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=6)
best_hyperparameters = study.best_params
print(best_hyperparameters)
```
Here below, is how we trained the model with the best parameters on the whole training dataset: 
```python
model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=len(label_encoder.classes_))
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=best_hyperparameters['per_device_train_batch_size'],
    per_device_eval_batch_size=16,
    warmup_steps=best_hyperparameters['warmup_steps'],
    weight_decay=best_hyperparameters['weight_decay'],
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print("Accuracy:", results['eval_accuracy'])
```

## Conclusion

EPFL_IKEA represents a significant step forward in applying AI to language learning. Our model not only facilitates better learning experiences by ensuring content appropriateness but also demonstrates the potential of AI in educational settings. We continue to refine our approach, aiming to broaden its application and assist an even larger audience of French language learners.

### Ranking

As a side note, we successfully placed in the top 10, finishing in 8th position!

<img width="1360" alt="Screenshot 2024-05-23 at 23 27 03" src="https://github.com/AlexPinel06/Team-IKEA-ML/assets/145569112/0a8add7a-f739-41af-b1ba-ac46f35c160e">


### Further Consideration 

To further enhance the effectiveness and accuracy of our model, we are considering several key improvements:

- Expanding the Dataset: Increasing the volume and variety of training data can help improve the model's robustness and its ability to generalize across a broader spectrum of text complexities. More data can particularly help in refining the nuances between closely rated difficulty levels.

- Diverse Language Sources: Integrating texts from a wider range of sources, including newspapers, literature, and academic articles, could provide a richer linguistic context, helping the model learn more complex patterns and styles of language use.

- User Feedback Integration: Incorporating user feedback into the training process can dynamically adapt the model to real-world effectiveness and user satisfaction, thereby continuously improving the prediction accuracy.

By focusing on these areas, we aim to make our tool even more effective and versatile, supporting a wider range of learners in achieving their language proficiency goals.
