# Fake News Classification Experiments

Two Python experiments for classifying labeled news articles: a feature-based logistic-regression baseline and a fine-tuned DistilBERT classifier.

## Approaches

| Script | Input | Actual model inputs |
| --- | --- | --- |
| `ML.py` | `Random_sampling.csv` (not included) | Article length, TextBlob sentiment, readability, named-entity count |
| `improved_ML.py` | Bundled `news.csv` | Tokenized, lemmatized/filtered article text |

The transformer script also computes engineered statistics, but those columns are **not passed into DistilBERT**. The baseline imports TF-IDF helpers but does not use TF-IDF for its final feature matrix.

## Setup

Use Python 3 in an isolated environment. The repository has no dependency manifest/lockfile; the following packages are inferred from imports.

```bash
git clone https://github.com/AbdelilahNossair/Fake-News-Detection.git
cd Fake-News-Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy textblob spacy textstat scikit-learn
python -m pip install torch transformers datasets accelerate
python -m spacy download en_core_web_sm
```

A compatible PyTorch/Transformers installation is required for Trainer. Dependency APIs have changed over time; if you encounter argument/import errors, establish and record a compatible version set. Internet access is needed to obtain the spaCy and DistilBERT models. GPU acceleration can substantially reduce training time.

## Data contract

The transformer script expects `news.csv` in the current directory with `text` and `label` columns. Labels map as follows:

```text
FAKE -> 0
REAL -> 1
```

Check missing text, unexpected labels, duplicate articles, and source overlap before training. The baseline requires the separate `Random_sampling.csv`; supply it with the same relevant columns or deliberately edit the input path. Do not assume the filename used by one script is used by both.

## Run the transformer experiment

```bash
python improved_ML.py
```

The workflow:

1. Loads spaCy and the CSV.
2. Lemmatizes/lowercases text and removes stop words/nonalphabetic tokens.
3. Computes auxiliary text statistics.
4. Tokenizes with `distilbert-base-uncased`, padding and truncating to the tokenizer's maximum length.
5. Makes an 80/20 split through Hugging Face Datasets.
6. Fine-tunes for three epochs with per-device training batch size 8, warmup steps 500, and weight decay 0.01.
7. Prints precision, recall, F1, accuracy, and a confusion matrix.

Training outputs/logging target `./results` and `./logs`. There is no separate inference service or explicit final-model export command in this script.

## Run the baseline

After preparing its expected CSV:

```bash
python ML.py
```

It uses an 80/20 scikit-learn split with random_state=42, fits logistic regression with max_iter=1000, and prints coefficients, accuracy, and a classification report. Coefficient interpretation depends on the actual class order and feature scaling.

## Interpretation and limitations

These are supervised dataset experiments, not an automated truth-verification system. Article style, corpus source, and topic can dominate predictions. The two scripts do not share identical data paths or a guaranteed identical split, so their printed scores are not a controlled head-to-head comparison.

The transformer split does not explicitly set a seed in the script. Lemmatization and stop-word removal also change the pretrained model's natural-text input distribution. No fixed environment, external-source evaluation, leakage audit, or reproducible benchmark result is supplied here.

A useful next step is one documented dataset, duplicate/source-aware splits, saved preprocessing/model artifacts, and a compact results table with seed and dependency versions. No standalone license file is included; dataset reuse rights must be checked separately.
