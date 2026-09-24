"""
app.py
Minimal Flask web form: paste an email -> get suspicious/not-suspicious + confidence.

Usage:
    python app.py
    then open http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request

from predict_cli import load_bundle, classify

app = Flask(__name__)
bundle = load_bundle()

PAGE = """
<!doctype html>
<title>Phishing Email Checker</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  textarea, input { width: 100%; box-sizing: border-box; margin-bottom: 10px; padding: 8px; }
  textarea { height: 160px; }
  button { padding: 10px 18px; cursor: pointer; }
  .result { margin-top: 20px; padding: 14px; border-radius: 8px; }
  .suspicious { background: #fde2e2; border: 1px solid #e57373; }
  .safe { background: #e2fde4; border: 1px solid #73c485; }
</style>
<h2>Phishing / Suspicious Email Checker</h2>
<form method="post">
  <label>Sender address</label>
  <input name="sender" value="{{ sender or '' }}">
  <label>Reply-To (optional)</label>
  <input name="reply_to" value="{{ reply_to or '' }}">
  <label>Subject</label>
  <input name="subject" value="{{ subject or '' }}">
  <label>Body</label>
  <textarea name="body">{{ body or '' }}</textarea>
  <button type="submit">Check email</button>
</form>
{% if verdict %}
<div class="result {{ 'suspicious' if verdict == 'SUSPICIOUS' else 'safe' }}">
  <strong>Verdict: {{ verdict }}</strong><br>
  Confidence phishing: {{ '%.1f' % (proba*100) }}%
</div>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def index():
    verdict = proba = None
    subject = sender = reply_to = body = ""
    if request.method == "POST":
        subject = request.form.get("subject", "")
        sender = request.form.get("sender", "")
        reply_to = request.form.get("reply_to", "")
        body = request.form.get("body", "")
        pred, proba = classify(bundle, subject, body, sender, reply_to)
        verdict = "SUSPICIOUS" if pred == 1 else "not suspicious"
    return render_template_string(
        PAGE, verdict=verdict, proba=proba,
        subject=subject, sender=sender, reply_to=reply_to, body=body,
    )


if __name__ == "__main__":
    app.run(debug=True)
