"""
features.py
Extracts hand-crafted features from raw email text for the phishing classifier.

The feature set combines:
  - TF-IDF over the email body/subject (captures wording patterns)
  - A handful of engineered signals known to correlate with phishing:
      * suspicious keyword hits ("verify your account", "urgent", etc.)
      * URL count, and whether any URL uses an IP address or a shortener
      * mismatched / suspicious sender domain patterns
      * excessive punctuation / all-caps ratio (social-engineering urgency cues)
      * presence of attachments-style keywords, request for credentials, etc.
"""

import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ---- Keyword lists-------------------------------

SUSPICIOUS_KEYWORDS = [
    "verify your account", "verify your identity", "update your payment",
    "urgent action required", "your account will be suspended",
    "click here", "confirm your password", "unusual sign-in activity",
    "you have won", "claim your prize", "wire transfer", "gift card",
    "tax refund", "invoice attached", "password expires", "act now",
    "limited time", "security alert", "reset your password",
    "confirm your identity", "unauthorized login",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorte.st",
}

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
    "icloud.com", "mail.com", "protonmail.com",
}

URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}")


def _domain_of(email_address: str) -> str:
    email_address = "" if email_address is None else str(email_address)
    if email_address.lower() == "nan":
        email_address = ""
    m = re.search(r"@([\w\.-]+)", email_address)
    return m.group(1).lower() if m else ""


def _extract_urls(text: str):
    return URL_RE.findall(text or "")


def engineered_features(subject: str, body: str, sender: str, reply_to: str = "") -> dict:
    """Compute a dict of hand-crafted numeric features for one email."""
    subject = "" if pd.isna(subject) else str(subject)
    body = "" if pd.isna(body) else str(body)
    text = f"{subject}\n{body}"
    text_lower = text.lower()

    urls = _extract_urls(text)
    def _safe_netloc(u: str) -> str:
        try:
         return urlparse(u).netloc.lower()
        except ValueError:
             return ""

    url_domains = [_safe_netloc(u) for u in urls]

    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text_lower)

    n_ip_urls = sum(1 for d in url_domains if IP_RE.match(d))
    n_shortened = sum(1 for d in url_domains if any(s in d for s in URL_SHORTENERS))

    sender_domain = _domain_of(sender)
    reply_domain = _domain_of(reply_to)
    sender_reply_mismatch = int(bool(reply_domain) and reply_domain != sender_domain)
    sender_is_freemail = int(sender_domain in FREEMAIL_DOMAINS)

    # domain mentioned in body text but different from sender domain -> spoof-ish
    mentioned_domains = set(url_domains)
    domain_mismatch = int(
        bool(sender_domain)
        and any(d and sender_domain not in d and d not in sender_domain for d in mentioned_domains)
    )

    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    exclaim_count = text.count("!")

    asks_for_credentials = int(
        any(k in text_lower for k in ["password", "ssn", "social security", "pin number", "login credentials"])
    )

    return {
        "n_urls": len(urls),
        "n_ip_urls": n_ip_urls,
        "n_shortened_urls": n_shortened,
        "keyword_hits": keyword_hits,
        "sender_reply_mismatch": sender_reply_mismatch,
        "sender_is_freemail": sender_is_freemail,
        "domain_mismatch": domain_mismatch,
        "caps_ratio": round(caps_ratio, 4),
        "exclaim_count": exclaim_count,
        "asks_for_credentials": asks_for_credentials,
        "body_length": len(body),
    }


ENGINEERED_FEATURE_NAMES = list(engineered_features("", "", "").keys())


class EmailFeaturizer:
    """
    Combines TF-IDF text features with engineered signal features into
    a single sparse feature matrix. Fit on training data, then reuse
    for transforming new emails (including a single email at inference time).
    """

    def __init__(self, max_tfidf_features: int = 3000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _texts_and_eng(self, df):
        texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).tolist()
        eng_rows = [
            engineered_features(
                row.get("subject", ""),
                row.get("body", ""),
                row.get("sender", ""),
                row.get("reply_to", ""),
            )
            for _, row in df.iterrows()
        ]
        eng_matrix = np.array([[r[k] for k in ENGINEERED_FEATURE_NAMES] for r in eng_rows], dtype=float)
        return texts, eng_matrix

    def fit_transform(self, df):
        texts, eng_matrix = self._texts_and_eng(df)
        tfidf = self.vectorizer.fit_transform(texts)
        eng_scaled = self.scaler.fit_transform(eng_matrix)
        self._fitted = True
        return hstack([tfidf, csr_matrix(eng_scaled)]).tocsr()

    def transform(self, df):
        if not self._fitted:
            raise RuntimeError("EmailFeaturizer must be fit before calling transform().")
        texts, eng_matrix = self._texts_and_eng(df)
        tfidf = self.vectorizer.transform(texts)
        eng_scaled = self.scaler.transform(eng_matrix)
        return hstack([tfidf, csr_matrix(eng_scaled)]).tocsr()

    def feature_names(self):
        return list(self.vectorizer.get_feature_names_out()) + ENGINEERED_FEATURE_NAMES
