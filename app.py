import os
import time
import requests
import secrets
import threading
import functools
import regex as re
from datetime import timedelta
from flask import Flask, render_template, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from tavily import TavilyClient
import logging
from flask_caching import Cache

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder=None)
app.secret_key = secrets.token_hex(24)
app.permanent_session_lifetime = timedelta(hours=2)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"]
)
limiter.init_app(app)

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
DOBBY_MODEL = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cache = Cache(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 600,
    'CACHE_THRESHOLD': 200
})

session_lock = threading.Lock()

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        if duration > 5:
            app.logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    return wrapper

@monitor_performance
def research_topic(topic):
    if not tavily_client:
        return "Search is disabled as TAVILY_API_KEY is not set."
    try:
        search_queries = [
            f"{topic} cryptocurrency project tokenomics roadmap whitepaper official documentation",
            f"{topic} crypto analysis use cases community updates"
        ]
        all_context = []
        for query in search_queries[:1]:
            try:
                response = tavily_client.search(
                    query=query, 
                    search_depth="advanced",
                    max_results=5
                )
                for res in response.get('results', []):
                    if len(res['content']) > 100:
                        all_context.append(f"- {res['content'][:300]}... (Source: {res['url']})")
                if len(all_context) >= 2:
                    break
            except Exception as e:
                app.logger.warning(f"Search query failed: {e}")
                continue
        return "\n".join(all_context) if all_context else "No relevant search results found."
    except Exception as e:
        app.logger.error(f"Tavily search error: {e}")
        return "Failed to fetch information due to search service error."

@cache.cached(timeout=600, key_prefix='trending_crypto')
@monitor_performance
def get_trending_crypto():
    try:
        headers = {"Accept": "application/json"}
        if COINGECKO_API_KEY:
            headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
        r = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        trending_data = []
        for coin in data.get("coins", [])[:5]:
            item = coin["item"]
            trending_data.append({
                "name": item["name"],
                "symbol": item["symbol"],
                "market_cap_rank": item.get("market_cap_rank", "N/A")
            })
        return trending_data if trending_data else []
    except Exception as e:
        app.logger.error(f"CoinGecko trending error: {e}")
        return []

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

def build_system_prompt(style, is_pro_mode, with_hashtags, topic):
    base_prompt = """You are Dobby, a professional content creator with deep expertise in crafting engaging, accurate, and impactful posts for X (Twitter).

CORE PRINCIPLES:
- Always provide accurate, fact-based information
- Never fabricate or speculate on unverified data
- Maintain a professional yet engaging tone
- Focus strictly on the specified topic
- Use proper grammar, spelling, and formatting
- Be concise but informative

WRITING GUIDELINES:
- Start with the most important information first
- Use clear, accessible language that both beginners and experts can understand
- Include specific details when available
- Avoid unnecessary jargon unless explaining technical concepts
- Structure information logically with smooth transitions"""

    if is_pro_mode:
        base_prompt += "\n\nMODE: Professional Analysis - Provide comprehensive, detailed analysis with:"
        base_prompt += "\n• Fundamentals and use cases"
        base_prompt += "\n• Position and competitive analysis"
        base_prompt += "\n• Risk factors and opportunities"
        base_prompt += "\n• Clear structure with introduction, analysis, and conclusion"
        base_prompt += "\n• Target length: 500-1000 words"
    else:
        base_prompt += "\n\nMODE: Social Media - Create engaging tweet content:"
        base_prompt += "\n• Maximum 280 characters including hashtags"
        base_prompt += "\n• Hook readers with compelling opening"
        base_prompt += "\n• Include key facts or insights"
        base_prompt += "\n• Use emojis sparingly but effectively"

    style_instructions = {
        "meme": "\n\nSTYLE: Witty & Memorable - Use clever wordplay, culture references, and humor while maintaining accuracy.",
        "education": "\n\nSTYLE: Educational & Clear - Break down complex concepts into digestible information.",
        "engagement": "\n\nSTYLE: Discussion-Driven - Pose thought-provoking questions or controversial statements that encourage replies.",
        "news": "\n\nSTYLE: News & Updates - Present information in a journalistic style with context and implications."
    }
    
    base_prompt += style_instructions.get(style, style_instructions["education"])

    if with_hashtags:
        base_prompt += "\n\nHASHTAGS: Include 1-3 relevant, popular hashtags that align with the content and community standards."
    else:
        base_prompt += "\n\nHASHTAGS: Do not include any hashtags in the response."

    base_prompt += f"\n\nTOPIC FOCUS: All content must directly relate to '{topic}' and provide value to readers interested in this specific subject."
    base_prompt += "\n\nOUTPUT LANGUAGE: English only. Ensure perfect grammar and professional presentation."

    return base_prompt

@monitor_performance
def call_dobby(system_prompt, user_message, temperature=0.7, is_pro_mode=False, num_outputs=1, chat_history=None):
    max_tokens = 4000 if is_pro_mode else 400
    temperature = min(temperature, 0.9)
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-4:])
    messages.append({"role": "user", "content": user_message})
    payload = {
        "model": DOBBY_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": 0.95,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "n": min(num_outputs, 3)
    }
    try:
        r = requests.post(FIREWORKS_URL, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("choices"):
            return ["No response available."]
        replies = []
        for choice in data["choices"]:
            content = choice["message"]["content"].strip()
            if content:
                content = re.sub(r'\n{3,}', '\n\n', content)
                content = re.sub(r' {2,}', ' ', content)
                if content and not is_pro_mode:
                    sentences = content.split('. ')
                    sentences = [s.strip().capitalize() if s else s for s in sentences]
                    content = '. '.join(sentences)
                if not is_pro_mode:
                    content = truncate_for_twitter(content)
                replies.append(content)
        return replies if replies else ["AI service returned empty response."]
    except Exception as e:
        app.logger.error(f"Fireworks API error: {e}")
        return ["AI service temporarily unavailable. Please try again."]

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

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/favicon.png")
def favicon_png():
    return "", 204

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
    if len(topic) > 200:
        return jsonify({"error": "Topic too long. Please limit to 200 characters."}), 400
    style = data.get("style", "education")
    is_pro_mode = bool(data.get("proMode", False))
    with_hashtags = bool(data.get("hashtags", True))
    try:
        temperature = float(data.get("temperature", 0.7))
        temperature = max(0.1, min(1.5, temperature))
    except:
        temperature = 0.7
    num_outputs = int(data.get("numOutputs", 1)) if "numOutputs" in data else 1
    num_outputs = min(num_outputs, 3)
    trending = get_trending_crypto()
    research_context = research_topic(topic)
    chat_history = session.get("chat_history", [])
    system_prompt = build_system_prompt(style, is_pro_mode, with_hashtags, topic)
    trending_text = ""
    if trending:
        trending_names = [t["name"] for t in trending[:3]]
        trending_text = f"Currently trending: {', '.join(trending_names)}"
    user_message = f"""TOPIC: {topic}

TRENDING CRYPTO:
{trending_text}

RESEARCH CONTEXT:
{research_context}

STYLE REQUESTED: {style}
PRO MODE: {'Yes' if is_pro_mode else 'No'}

Please create content about {topic} following the guidelines above."""
    replies = call_dobby(system_prompt, user_message, temperature, is_pro_mode, num_outputs, chat_history)
    quality_replies = []
    for reply in replies:
        if len(reply.strip()) > 10 and not reply.lower().startswith("i'm sorry") and "unavailable" not in reply.lower():
            quality_replies.append(reply)
    if not quality_replies:
        quality_replies = ["Unable to generate quality content. Please try rephrasing your topic or adjusting parameters."]
    if quality_replies:
        update_chat_history(topic, quality_replies[0])
    return jsonify({
        "replies": quality_replies,
        "trending": trending,
        "ts": int(time.time())
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ts": int(time.time())})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def rate_limit(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    application = app
