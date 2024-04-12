import pandas as pd
from textblob import TextBlob
import spacy
import textstat
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from transformers import EvalPrediction
import numpy as np

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv('news.csv')
print("1")

# Define a mapping from label strings to integers
label_to_int = {"FAKE": 0, "REAL": 1}

# Apply this mapping to your labels
df['label'] = df['label'].map(label_to_int)
print("1.1")


# Preprocessing function for NER
def preprocess_text_for_ner(text):
    doc = nlp(text)
    preprocessed_text = ' '.join([token.lemma_.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha])
    return preprocessed_text

df['preprocessed_text'] = df['text'].apply(preprocess_text_for_ner)

# Additional text analyses
df['article_length'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
df['sentiment'] = df['preprocessed_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['readability'] = df['preprocessed_text'].apply(lambda x: textstat.flesch_reading_ease(x))
df['entities_count'] = df['preprocessed_text'].apply(lambda x: len(nlp(x).ents))
print("2")


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['preprocessed_text'], padding="max_length", truncation=True)


hf_dataset = Dataset.from_pandas(df[['preprocessed_text', 'label']])
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
train_test_dataset = tokenized_dataset.train_test_split(test_size=0.2)
print("3")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
print("3")
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    precision = precision_score(p.label_ids, preds, average='binary')
    recall = recall_score(p.label_ids, preds, average='binary')
    f1 = f1_score(p.label_ids, preds, average='binary')
    accuracy = accuracy_score(p.label_ids, preds)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


# Include the compute_metrics function in your Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_dataset['train'],
    eval_dataset=train_test_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

print("Test set metrics:", results)

#Obtain Predictions
predictions = trainer.predict(train_test_dataset['test'])

# 'predictions.predictions' contains the logits or output scores from the model
# We apply softmax to convert these logits to probabilities and then argmax to get the predicted class
preds = np.argmax(predictions.predictions, axis=-1)

# 'predictions.label_ids' contains the true labels
y_true = predictions.label_ids
y_pred = preds

# Step 2: Calculate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
