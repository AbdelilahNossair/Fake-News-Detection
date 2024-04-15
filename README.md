# Fake News Detection Project

## Overview:
  This project aims to detect fake news using advanced NLP techniques and deep learning models. 
  We leverage the power of spaCy for natural language processing, TextBlob for sentiment analysis, 
  textstat for readability scoring, and the Hugging Face Transformers library for sequence classification with a DistilBert model.

## Requirements:
 - pandas
  - TextBlob
  - spaCy
  - textstat
  - Hugging Face Transformers
  - datasets library
  - NumPy
  - scikit-learn

## Installation:  
```bash
  pip install pandas textblob spacy textstat transformers datasets numpy scikit-learn
  python -m spacy download en_core_web_sm
```

## Dataset:
  We use a dataset named news.csv, which contains labeled news articles as either "FAKE" or "REAL".

## Preprocessing:
  Text Cleaning and Lemmatization: We clean the text by removing stop words and non-alphabetic characters, and then lemmatize the text using spaCy.
  Feature Engineering: We add features such as article length, sentiment polarity, readability score, and entities count to the dataset.

## Model Training:
  We use the DistilBert model from the Hugging Face Transformers library for sequence classification.
  The model is fine-tuned on our dataset to classify news articles as fake or real.

## Training Process:
  Tokenization: The text data is tokenized using DistilBertTokenizerFast from the Hugging Face library.
  Dataset Splitting: The dataset is split into training and testing sets with an 80/20 ratio.
  Model Configuration and Training: We configure the training arguments and initiate training using the Trainer API from Hugging Face. 
        The model is evaluated on the test set post-training.

## Evaluation:
  The model's performance is evaluated based on accuracy on the test set. This metric helps us understand how well our model can generalize to unseen data.

## Usage:
Follow the steps in the provided code snippets to run the training process, evaluate the model, and obtain predictions. Ensure you have the necessary libraries installed and the dataset ready.

### Step 1: Define the `compute_metrics` Function
Define the `compute_metrics` function to calculate precision, recall, F1 score, and accuracy. This function will be used by the `Trainer` to compute metrics on the evaluation set.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
from transformers import EvalPrediction

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
```

### Step 2: Include the `compute_metrics` Function in Your Trainer Setup
Instantiate the `Trainer` with the `compute_metrics` function. Train your model and evaluate it using the `Trainer`.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_dataset['train'],
    eval_dataset=train_test_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("Test set metrics:", results)
```

### Step 3: Obtain Predictions and Calculate Confusion Matrix
After training and evaluation, obtain predictions for the test dataset and calculate the confusion matrix to analyze the model's performance further.

```python
predictions = trainer.predict(train_test_dataset['test'])
preds = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
y_pred = preds

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
```

Follow these steps to train your model, evaluate its performance, obtain predictions, and understand its performance in detail through various metrics and the confusion matrix. Ensure all necessary Python libraries for computation and evaluation are installed.

## Contribution: 
  We welcome contributions to improve the model's accuracy and efficiency or extend its capabilities. Please feel free to submit pull requests or open issues to discuss potential improvements.
