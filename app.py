import os
import time
import requests
import secrets
import threading
import functools
import regex as re
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from tavily import TavilyClient
import logging
from flask_caching import Cache

load_dotenv()

app = Flask(__name__)

app.secret_key = secrets.token_hex(24)
app.permanent_session_lifetime = timedelta(hours=2)

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"]
)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
DOBBY_MODEL = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_THRESHOLD': 100
})

session_lock = threading.Lock()

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if duration > 5:
                app.logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            app.logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    return wrapper

def filter_profanity(text):
    profane_words = {
        "bastard", "dick", "pussy", "scoundrel", "dog", "fuck",
        "fucking", "shit", "bitch", "cunt", "asshole"
    }
    words = text.split()
    filtered_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word not in profane_words:
            filtered_words.append(word)
    return " ".join(filtered_words)

@monitor_performance
def research_topic(topic):
    if not tavily_client:
        return "Search is disabled as TAVILY_API_KEY is not set."
    try:
        search_query = f"in-depth technical details, use cases, and tokenomics of crypto project '{topic}'"
        response = tavily_client.search(query=search_query, search_depth="advanced", max_results=5)
        context = "\n".join([f"- {res['content']} (Source: {res['url']})" for res in response.get('results', [])])
        return context if context else "No relevant search results found."
    except Exception as e:
        app.logger.error(f"Tavily search error: {e}")
        return "Failed to fetch information due to search service error."

@cache.cached(timeout=300, key_prefix='trending_crypto')
@monitor_performance
def get_trending_crypto():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
            headers={"Accept": "application/json"}
        )
        r.raise_for_status()
        data = r.json()
        names = [c["item"]["name"] for c in data.get("coins", [])][:7]
        return ", ".join(names) if names else "None"
    except Exception as e:
        app.logger.error(f"CoinGecko error: {e}")
        return "Could not fetch trending data"

def get_crypto_price(topic):
    topic_id = re.sub(r'[^a-zA-Z0-9\-]', '', topic.lower().replace(" ", "-"))
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={topic_id}&vs_currencies=usd",
            timeout=6,
            headers={"Accept": "application/json"}
        )
        r.raise_for_status()
        data = r.json()
        price = data.get(topic_id, {}).get("usd")
        return f"Current price: ${price}" if price else "Price not available"
    except Exception:
        return "Price not available"

def calculate_tweet_length(text):
    url_pattern = re.compile(r'https?://[^\s]+')
    urls = url_pattern.findall(text)
    text_without_urls = url_pattern.sub('', text)
    char_count = len(text_without_urls)
    url_count = len(urls) * 23
    return char_count + url_count

def truncate_for_twitter(text):
    if calculate_tweet_length(text) <= 280:
        return text
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        if calculate_tweet_length(text[:mid]) <= 277:
            left = mid
        else:
            right = mid - 1
    return text[:left].rstrip() + "..."

def build_system_prompt(style, is_pro_mode, with_hashtags):
    lines = [
        "You are Dobby, a sharp, witty, and crypto-native copywriter for Twitter from the 'unhinged' series.",
        "You sound like a human, not a corporate bot. You are deeply familiar with crypto culture and community inside jokes.",
        "You must focus STRICTLY on the user's main topic.",
        "Keep names, tickers, and facts consistent with the provided context.",
        "Do NOT fabricate news or data.",
        "Output language: English only."
    ]
    if is_pro_mode:
        lines.append("PRO MODE: Generate a comprehensive, detailed, and in-depth answer.")
        lines.append("Tone: Professional, analytical, and objective.")
    else:
        lines.append("STANDARD MODE: Concise tweet under 280 chars.")
    if style == "meme":
        lines.append("Style: Meme-grade humor on the topic only.")
    elif style == "education":
        lines.append("Style: Educational, breaking down tech/use cases/tokenomics.")
    elif style == "engagement":
        lines.append("Style: Ask an open-ended engagement question.")
    elif style == "news":
        lines.append("Style: Concise news summary about the topic.")
    if with_hashtags:
        lines.append("Always include 1-2 relevant hashtags.")
    else:
        lines.append("Do NOT include hashtags.")
    return "\n".join(lines)

@monitor_performance
def call_dobby(system_prompt, user_message, temperature=0.8, is_pro_mode=False, num_outputs=1, chat_history=None):
    max_tokens = 3000 if is_pro_mode else 290
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})
    payload = {
        "model": DOBBY_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "n": num_outputs
    }
    try:
        r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("choices"):
            return ["No response available."]
        replies = [c["message"]["content"].strip() for c in data["choices"]]
        return [truncate_for_twitter(r) if not is_pro_mode else r for r in replies]
    except Exception as e:
        app.logger.error(f"Fireworks error: {e}")
        return ["AI service unavailable."]

def update_chat_history(user_msg, bot_reply):
    with session_lock:
        history = session.get("chat_history", [])
        history = (history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_reply}
        ])[-6:]
        session["chat_history"] = history
        session.modified = True

@app.route("/", methods=["GET"])
def home():
    session.permanent = True
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("15 per minute")
def chat():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    data = request.get_json()
    if not FIREWORKS_API_KEY:
        return jsonify({"error": "FIREWORKS_API_KEY not set"}), 500
    topic = (data.get("message") or "").strip()
    if not topic:
        return jsonify({"error": "Message cannot be empty"}), 400
    style = data.get("style", "meme")
    is_pro_mode = bool(data.get("proMode", False))
    with_hashtags = bool(data.get("hashtags", False))
    try:
        temperature = float(data.get("temperature", 0.8))
        temperature = max(0.1, min(2.0, temperature))
    except:
        temperature = 0.8
    num_outputs = int(data.get("numOutputs", 1)) if "numOutputs" in data else 1
    trending = get_trending_crypto()
    research_context = ""
    price_info = get_crypto_price(topic)
    if style in ["education", "news"] or (style == "engagement" and is_pro_mode):
        research_context = research_topic(topic)
    chat_history = session.get("chat_history", [])
    system_prompt = build_system_prompt(style, is_pro_mode, with_hashtags)
    user_message = (
        f"Topic: {topic}\n"
        f"Style: {style}\n"
        f"Context:\n{research_context}\n\n"
        f"Extra: {price_info}\nTrending: {trending}\n"
    )
    replies = call_dobby(system_prompt, user_message, temperature, is_pro_mode, num_outputs, chat_history)
    replies = [filter_profanity(r) for r in replies]
    if replies:
        update_chat_history(topic, replies[0])
    return jsonify({"replies": replies, "trending": trending, "ts": int(time.time())})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ts": int(time.time())})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def rate_limit(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    application = app
