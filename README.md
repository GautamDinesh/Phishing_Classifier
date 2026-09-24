# Phishing / Suspicious Email Classifier

A lightweight NLP pipeline that flags suspicious emails, combining TF-IDF text
features with hand-engineered phishing signals (URL patterns, sender/domain
mismatches, urgency language, credential requests, etc.).

## Project layout

```
phishing_classifier/
├── data/
│   └── emails.csv          # subject, body, sender, reply_to, label (1=phishing, 0=legit)
├── model/
│   └── phishing_model.joblib   # saved after training
├── features.py              # feature extraction (TF-IDF + engineered signals)
├── make_sample_data.py      # generates a small demo dataset
├── train.py                 # trains + evaluates the classifier
├── predict_cli.py           # CLI: paste an email, get a verdict
├── app.py                   # optional Flask web form
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get a dataset

A small synthetic sample is included to get you running end-to-end:

```bash
python make_sample_data.py     # writes data/emails.csv
```

For real results, swap this out for a real dataset with the same columns
(`subject, body, sender, reply_to, label`). Good public sources:

- Kaggle "Phishing Email Dataset" / "Nazario phishing corpus"
- Enron email dataset (as the "legit" / ham class)

Just reshape whatever you download into a CSV with those five columns.

## 2. Train

```bash
python train.py --data data/emails.csv --model logreg   # or --model rf
```

Prints a classification report, confusion matrix, ROC AUC, and (for logreg)
the top phishing-indicative features. Saves the fitted model to
`model/phishing_model.joblib`.

## 3. Classify emails

CLI, interactive:
```bash
python predict_cli.py
```

CLI, one-shot / scriptable:
```bash
python predict_cli.py --subject "Verify your account" \
  --sender "security@paypa1-support.com" \
  --body "Click here to verify your identity within 24 hours: http://192.168.4.22/verify"
```

Web form:
```bash
python app.py
# open http://127.0.0.1:5000
```

## How it works

- **Text features**: TF-IDF over subject + body (unigrams + bigrams).
- **Engineered features** (`features.py`):
  - suspicious keyword hits ("verify your account", "urgent", "gift card", ...)
  - URL count, IP-address URLs, shortened-URL usage
  - sender/reply-to domain mismatch, sender on a free-mail domain
  - a URL domain mentioned in the body that doesn't match the sender's domain
  - ALL-CAPS ratio and exclamation-mark count (urgency/social-engineering cues)
  - whether the email asks for credentials (password, SSN, PIN, etc.)
- **Model**: Logistic Regression (interpretable, shows top coefficients) or
  Random Forest, both with `class_weight="balanced"` since phishing datasets
  are often imbalanced.

## Extending this

- Add sender-reputation lookups (SPF/DKIM pass-fail, domain age via WHOIS).
- Add attachment-type flags (.exe, .scr, macro-enabled Office docs).
- Try a transformer-based text encoder instead of TF-IDF for better recall
  on novel phishing wording.
- Track false positives/negatives over time and retrain periodically.
