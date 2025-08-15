import os
import time
import requests
import secrets
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from tavily import TavilyClient
import logging
from flask_caching import Cache

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
DOBBY_MODEL = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

def filter_profanity(text):
    profane_words = {
        "dog", "bastard", "dick", "pussy", "scoundrel", "dog", 
        "fuck", "fucking", "shit", "bitch", "cunt", "asshole", "dick", "pussy"
    }
    words = text.split()
    filtered_words = []
    for word in words:
        cleaned = word.lower().strip(".,!?")
        if any(p in cleaned for p in profane_words):
            filtered_words.append("*" * len(word))
        else:
            filtered_words.append(word)
    return " ".join(filtered_words)

def research_topic(topic):
    if not TAVILY_API_KEY:
        return "Search is disabled as TAVILY_API_KEY is not set."
    
    try:
        search_query = f"in-depth technical details, use cases, and tokenomics of crypto project '{topic}'"
        response = tavily_client.search(query=search_query, search_depth="basic", max_results=5)
        
        context = "\n".join([f"- {res['content']} (Source: {res['url']})" for res in response.get('results', [])])
        return context if context else "No relevant search results found."
    except Exception as e:
        logging.error(f"Tavily search error: {e}")
        return "Failed to fetch information."

@cache.cached(timeout=300, key_prefix='trending_crypto')
def get_trending_crypto():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=6,
            headers={"Accept": "application/json"}
        )
        r.raise_for_status()
        data = r.json()
        names = [c["item"]["name"] for c in data.get("coins", [])][:7]
        return ", ".join(names) if names else "None"
    except Exception as e:
        logging.error(f"CoinGecko error: {e}")
        return "None"

def get_crypto_price(topic):
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids={topic.lower()}&vs_currencies=usd",
            timeout=6,
            headers={"Accept": "application/json"}
        )
        r.raise_for_status()
        data = r.json()
        price = data.get(topic.lower(), {}).get("usd")
        return f"Current price: ${price}" if price else "Price not available"
    except Exception:
        return "Price not available"

def build_system_prompt(style, is_pro_mode, with_hashtags):
    lines = [
        "You are Dobby, a sharp, witty, and crypto-native copywriter for Twitter from the 'unhinged' series.",
        "You sound like a human, not a corporate bot. You are deeply familiar with crypto culture and community inside jokes.",
        "You must focus STRICTLY on the user's main topic. The 'trending coins' list is for background awareness only, not for creating content.",
        "Keep names, tickers, and facts consistent with the provided context.",
        "Do NOT fabricate news or data.",
        "Output language: English only. Do not use any other language in the response."
    ]
    
    if is_pro_mode:
        lines.append("PRO MODE: Generate a comprehensive, detailed, and in-depth answer based on the provided context. There is no character limit.")
    else:
        lines.append("STANDARD MODE: Generate a concise, sharp tweet. Target length is under 280 characters. Never exceed 280 characters.")

    if is_pro_mode:
        lines.append("Writing Tone: Professional, analytical, and objective. Suitable for serious investors.")

    if style == "meme":
        lines.append("Style: Witty and relatable meme-grade humor about the user's specific topic. Reference a known crypto meme or feeling related to the topic. Do not talk about other coins.")
    elif style == "education":
        lines.append("Style: Educational. Use the rich context from the web search to provide a detailed and insightful explanation. Break down the technology, use case, or tokenomics into key points.")
    elif style == "engagement":
        lines.append("Style: Community engagement. Ask an open-ended question about the user's topic that sparks genuine discussion or debate, not a simple yes/no question.")
    elif style == "news":
        lines.append("Style: News summary. Provide a concise summary of recent news or updates about the topic, based on provided context.")

    if with_hashtags:
        lines.append("Always include 1-2 relevant hashtags (cashtags like $BTC are preferred).")
    else:
        lines.append("Do NOT include hashtags.")

    return "\n".join(lines)

def call_dobby(system_prompt, user_message, temperature=0.8, is_pro_mode=False):
    max_tokens = 3000 if is_pro_mode else 280
    
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DOBBY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *session.get("chat_history", []),
            {"role": "user", "content": user_message}
        ],
        "max_tokens": max_tokens,
        "temperature": float(temperature)
    }
    try:
        r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        if not is_pro_mode and len(content) > 280:
            content = content[:277] + "..."
        return content
    except requests.HTTPError as e:
        logging.error(f"Fireworks API error: {e}")
        raise

@app.route("/", methods=["GET"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    topic = (data.get("message") or "").strip()
    style = data.get("style") or "meme"
    is_pro_mode = bool(data.get("proMode", False))
    with_hashtags = bool(data.get("hashtags", False))
    temperature = data.get("temperature", 0.8)

    if not FIREWORKS_API_KEY:
        return jsonify({"error": "FIREWORKS_API_KEY not set"}), 500
    if not topic:
        return jsonify({"reply": "Please provide a clear topic to write about."})

    trending = get_trending_crypto()
    
    research_context = ""
    price_info = get_crypto_price(topic)
    if style in ["education", "news"] or (style == "engagement" and is_pro_mode):
        research_context = research_topic(topic)

    system_prompt = build_system_prompt(style, is_pro_mode, with_hashtags)

    user_message = (
        f"Topic: {topic}\n"
        f"Style: {style}\n"
        f"Context from web search (use this for your answer):\n{research_context}\n\n"
        f"Additional data: {price_info}\n"
        f"Trending coins (for awareness only, do not create content about them unless the main topic is one of them): {trending}\n"
    )

    try:
        reply = call_dobby(system_prompt, user_message, temperature=temperature, is_pro_mode=is_pro_mode)
        filtered_reply = filter_profanity(reply)
    except Exception as e:
        filtered_reply = "An error occurred. Please try again later."

    history = session.get("chat_history", [])
    history = (history + [
        {"role": "user", "content": topic},
        {"role": "assistant", "content": filtered_reply}
    ])[-6:]
    session["chat_history"] = history
    session.modified = True

    return jsonify({"reply": filtered_reply, "trending": trending, "ts": int(time.time())})

if __name__ == "__main__":
    app.run(debug=False)