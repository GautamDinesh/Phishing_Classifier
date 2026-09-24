"""
make_sample_data.py
Generates a small labeled sample dataset (data/emails.csv) so the pipeline
runs end-to-end out of the box. This is NOT a substitute for a real dataset --
swap it out with something like:

  - Kaggle "Phishing Email Dataset" (https://www.kaggle.com/datasets)
  - Nazario phishing corpus + Enron ham corpus

as long as the final CSV has columns: subject, body, sender, reply_to, label
(label = 1 for phishing/suspicious, 0 for legitimate).
"""

import csv
import random

random.seed(42)

PHISHING_TEMPLATES = [
    ("Urgent: Verify your account now",
     "Dear customer, we detected unusual sign-in activity. Verify your identity "
     "within 24 hours or your account will be suspended. Click here: http://192.168.4.22/verify",
     "security@paypa1-support.com", "no-reply@totallylegit.ru"),
    ("Your invoice is overdue - action required",
     "Please confirm your password and billing details immediately to avoid "
     "service interruption. Wire transfer confirmation attached. Act now!",
     "billing@invoice-alerts.com", ""),
    ("You have won a $1000 gift card!!!",
     "Congratulations!! Claim your prize now by clicking the link and entering "
     "your login credentials: http://bit.ly/claim-prize-now",
     "promo@rewards-center.net", "winner@rewardz-claim.com"),
    ("Password expires today",
     "Your password expires today. Reset your password immediately using this "
     "secure link: http://account-verify-secure.com/reset?id=8827",
     "it-support@company-secure-login.com", ""),
    ("Unauthorized login attempt detected",
     "SECURITY ALERT: We noticed an unauthorized login from a new device. "
     "Confirm your identity now or your account will be locked: http://45.33.12.9/login",
     "alert@bank-secure-verify.com", "support@totally-not-a-bank.com"),
]

LEGIT_TEMPLATES = [
    ("Q3 project sync notes",
     "Hi team, attaching notes from today's sync. Let's follow up on the open "
     "action items before Friday's review. Thanks!",
     "maria.chen@acmecorp.com", ""),
    ("Lunch on Thursday?",
     "Hey, are you free for lunch Thursday around 12:30? There's a new place "
     "near the office I've been wanting to try.",
     "james.oliver@gmail.com", ""),
    ("Re: Budget spreadsheet",
     "Thanks for sending this over. I've reviewed the numbers and they look "
     "correct. I'll forward to finance for sign-off.",
     "priya.nair@acmecorp.com", "priya.nair@acmecorp.com"),
    ("Weekly newsletter - Engineering blog",
     "This week: our migration to the new build system, a retrospective on "
     "the Q2 outage, and three articles from the team.",
     "newsletter@engineeringweekly.com", ""),
    ("Meeting rescheduled to 3pm",
     "Quick note that our 1:1 has moved to 3pm today, same link. Let me know "
     "if that doesn't work.",
     "david.kim@acmecorp.com", "david.kim@acmecorp.com"),
]


def jitter(text: str) -> str:
    """Add tiny variation so repeated templates aren't identical rows."""
    suffixes = ["", " Thanks.", " Regards.", " Please advise.", " Let me know.",
                " Kind regards.", " Cheers.", " Best.", " Talk soon."]
    prefixes = ["", "Hi, ", "Hello, ", "Hey, ", "Note: ", "FYI - "]
    return random.choice(prefixes) + text + random.choice(suffixes)


def generate(n_per_class: int = 40):
    rows = []
    for i in range(n_per_class):
        subj, body, sender, reply_to = random.choice(PHISHING_TEMPLATES)
        rows.append([jitter(subj), jitter(body), sender, reply_to, 1])
    for i in range(n_per_class):
        subj, body, sender, reply_to = random.choice(LEGIT_TEMPLATES)
        rows.append([jitter(subj), jitter(body), sender, reply_to, 0])
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate(n_per_class=60)
    with open("data/emails.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "body", "sender", "reply_to", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to data/emails.csv")
