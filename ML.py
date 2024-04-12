import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy
import textstat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")


def preprocess_text_for_ner(text):
    
    #Preprocess text for NER by lemmatization, lowercasing, and removing stop words and non-alphabetic tokens.
    
    doc = nlp(text)
    preprocessed_text = ' '.join([token.lemma_.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha])
    return preprocessed_text

# Load dataset
df = pd.read_csv('Random_sampling.csv')
print("1")


# Apply preprocessing to each article's text
df['preprocessed_text'] = df['text'].apply(preprocess_text_for_ner)
print("2")


# Now df['preprocessed_text'] contains the preprocessed text suitable for NER and further analysis
df.head()



# Named Entity Recognition setup
def extract_entities(text):
    doc = nlp(text)
    return len(doc.ents)
print("3")


# Applying text analysis methods
df['article_length'] = df['text'].apply(lambda x: len(x.split()))
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['readability'] = df['text'].apply(lambda x: textstat.flesch_reading_ease(x))
df['entities_count'] = df['text'].apply(extract_entities)
print("4")


# Assuming a basic feature set for demonstration
features = ['article_length', 'sentiment', 'readability', 'entities_count']
X = df[features]
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Estimation: Model coefficients give us the estimated parameters
print(f"Model coefficients: {model.coef_}")

# Inference: Higher positive values indicate a stronger influence towards classifying an article as real
features_coef = pd.DataFrame(data=model.coef_.flatten(), index=features, columns=['Coefficient'])
print(features_coef)

# Making predictions on the test set
predictions = model.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")
print(classification_report(y_test, predictions))


