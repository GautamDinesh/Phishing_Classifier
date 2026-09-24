"""
predict_cli.py
Paste (or pipe) an email in, get a suspicious/not-suspicious verdict + confidence.

Usage:
    python predict_cli.py
        -> interactive prompt for subject / sender / reply-to / body

    python predict_cli.py --subject "..." --sender "..." --body "..."
        -> one-shot mode, good for scripting

    cat email.txt | python predict_cli.py --body -
        -> reads body from stdin
"""

import argparse
import sys

import joblib
import pandas as pd

MODEL_PATH = "model/phishing_model.joblib"


def load_bundle():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"No trained model found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)


def classify(bundle, subject, body, sender, reply_to=""):
    featurizer = bundle["featurizer"]
    model = bundle["model"]
    df = pd.DataFrame([{
        "subject": subject, "body": body, "sender": sender, "reply_to": reply_to,
    }])
    X = featurizer.transform(df)
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return pred, proba[1]  # proba[1] = P(phishing)


def interactive():
    print("Paste email details (press Enter to skip a field).\n")
    subject = input("Subject: ").strip()
    sender = input("Sender address: ").strip()
    reply_to = input("Reply-To address (if any): ").strip()
    print("Body (end with a blank line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    body = "\n".join(lines)
    return subject, body, sender, reply_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None)
    parser.add_argument("--sender", default="")
    parser.add_argument("--reply-to", default="")
    parser.add_argument("--body", default=None, help="Email body text, or '-' to read from stdin")
    args = parser.parse_args()

    bundle = load_bundle()

    if args.subject is None and args.body is None:
        subject, body, sender, reply_to = interactive()
    else:
        subject = args.subject or ""
        sender = args.sender
        reply_to = args.reply_to
        body = sys.stdin.read() if args.body == "-" else (args.body or "")

    pred, proba_phish = classify(bundle, subject, body, sender, reply_to)

    verdict = "SUSPICIOUS" if pred == 1 else "not suspicious"
    print(f"\nVerdict: {verdict}")
    print(f"Confidence phishing: {proba_phish:.1%}")
    print(f"Confidence legit:    {1 - proba_phish:.1%}")

    if pred == 1:
        print("\nReasons this may be flagged include suspicious links, urgent/")
        print("credential-request language, or sender/domain mismatches.")


if __name__ == "__main__":
    main()
