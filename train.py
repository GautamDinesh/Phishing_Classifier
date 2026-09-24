"""
train.py
Trains a phishing/suspicious-email classifier on data/emails.csv and saves
the fitted featurizer + model to model/.

"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import EmailFeaturizer


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "body", "sender", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if "reply_to" not in df.columns:
        df["reply_to"] = ""
    df[["subject", "body", "sender", "reply_to"]] = df[["subject", "body", "sender", "reply_to"]].fillna("")
    return df


def build_model(kind: str):
    if kind == "logreg":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    "change below line to training data file name"
    parser.add_argument("--data", default="data/CEAS_08.csv")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    df = load_data(args.data)
    y = df["label"].astype(int)

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=args.test_size, random_state=42, stratify=y
    )

    featurizer = EmailFeaturizer()
    X_train = featurizer.fit_transform(df_train)
    X_test = featurizer.transform(df_test)

    clf = build_model(args.model)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n=== Classification report ===")
    print(
        classification_report(
            y_test, y_pred, target_names=["legit", "phishing"]
        )
    )

    print("=== Confusion matrix ===")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["actual_legit", "actual_phishing"],
            columns=["pred_legit", "pred_phishing"],
        )
    )

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC: {auc:.3f}")
    except ValueError:
        pass  # only one class present in a tiny test split

    if args.model == "logreg" and hasattr(clf, "coef_"):
        names = featurizer.feature_names()
        coefs = clf.coef_[0]
        top_idx = coefs.argsort()[-15:][::-1]
        print("\nTop phishing-indicative features:")
        for i in top_idx:
            print(f"  {names[i]:30s} {coefs[i]:+.3f}")

    joblib.dump(
        {"featurizer": featurizer, "model": clf, "model_kind": args.model},
        "model/phishing_model.joblib",
    )
    print("\nSaved model to model/phishing_model.joblib")


if __name__ == "__main__":
    main()
