import os
import secrets
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__, template_folder="templates", static_folder=None)
app.secret_key = secrets.token_hex(24)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    application = app
