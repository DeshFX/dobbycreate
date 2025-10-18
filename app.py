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
import tweepy

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder=None)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(24))
app.permanent_session_lifetime = timedelta(hours=2)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour", "30 per minute"]
)
limiter.init_app(app)

# API Configuration
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
DOBBY_MODEL = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
try:
    tweepy_client = tweepy.Client(TWITTER_BEARER_TOKEN) if TWITTER_BEARER_TOKEN else None
except Exception as e:
    app.logger.error(f"Tweepy initialization failed: {e}")
    tweepy_client = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if duration > 5:
                app.logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            app.logger.error(f"{func.__name__} error: {str(e)}")
            raise
    return wrapper

@cache.cached(timeout=300, key_prefix='twitter_sentiment')
@monitor_performance
def get_twitter_sentiment(topic):
    if not tweepy_client:
        return "Twitter integration disabled (TWITTER_BEARER_TOKEN not set)."
    try:
        query = f'"{topic}" lang:en -is:retweet'
        response = tweepy_client.search_recent_tweets(
            query=query,
            max_results=10,
            tweet_fields=["text"]
        )
        if not response.data:
            return "No recent tweets found."
        
        tweets_text = [
            tweet.text.replace('\n', ' ').strip()
            for tweet in response.data
        ]
        return "\n".join(f"- {t}" for t in tweets_text)
    except Exception as e:
        app.logger.error(f"Twitter API error: {e}")
        return "Failed to fetch Twitter data."

@monitor_performance
def research_topic(topic):
    if not tavily_client:
        return "Search disabled (TAVILY_API_KEY not set)."
    try:
        search_queries = [
            f"{topic} cryptocurrency project tokenomics roadmap whitepaper",
            f"{topic} crypto analysis use cases community"
        ]
        
        all_context = []
        for query in search_queries:
            try:
                response = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5
                )
                
                for res in response.get('results', []):
                    if len(res.get('content', '')) > 100:
                        snippet = res['content'][:250]
                        all_context.append(f"{snippet}... (Source: {res['url']})")
                
                if len(all_context) >= 3:
                    break
            except Exception as e:
                app.logger.warning(f"Search query failed: {e}")
                continue
        
        return "\n".join(all_context) if all_context else "No research results found."
    except Exception as e:
        app.logger.error(f"Tavily error: {e}")
        return "Research service unavailable."

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
        return trending_data
    except Exception as e:
        app.logger.error(f"CoinGecko error: {e}")
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
    base_prompt = """You are Dobby, a professional AI content creator specializing in cryptocurrency analysis and insights.

CORE PRINCIPLES:
- Provide accurate, fact-based information only
- Never fabricate or speculate on unverified data
- Maintain professional yet engaging tone
- Structure content clearly with proper paragraphs
- Use natural language flow, avoid AI clichés
- Be concise but informative

WRITING GUIDELINES:
- Start with the most important information
- Use clear, accessible language for all knowledge levels
- Include specific details and metrics when available
- Avoid unnecessary jargon; explain technical concepts clearly
- Write in proper paragraphs with smooth transitions
- Minimize use of dashes and fragments
- Keep formatting consistent and scannable"""

    if is_pro_mode:
        base_prompt += """

CONTENT MODE: Professional Analysis (500-1000 words)
Structure your response as follows:

1. INTRODUCTION (1-2 paragraphs)
Overview of the topic with key context and significance.

2. FUNDAMENTALS (2-3 paragraphs)
Core features, technology, and infrastructure. Include:
- Technical architecture
- Key differentiators
- Ecosystem positioning

3. USE CASES & VALUE PROPOSITION (2-3 paragraphs)
Practical applications and why it matters. Discuss:
- Primary use cases
- Target audience
- Real-world benefits

4. COMPETITIVE ANALYSIS (1-2 paragraphs)
How it compares to alternatives:
- Key competitors
- Competitive advantages
- Market positioning

5. RISK FACTORS & OPPORTUNITIES (1-2 paragraphs)
Balanced perspective including:
- Potential risks
- Growth opportunities
- Timeline considerations

6. CONCLUSION (1-2 paragraphs)
Summary and investment perspective."""
    else:
        base_prompt += """

CONTENT MODE: Social Media Tweet (Max 280 characters)
- Hook readers immediately with compelling opening
- Include key insight or fact
- Use emojis sparingly (1-2 maximum)
- Create engagement or curiosity"""

    style_instructions = {
        "meme": "\n\nSTYLE: Entertaining & Witty\n- Use clever wordplay and cultural references\n- Keep tone playful and memorable\n- Maintain accuracy while being fun\n- NO profanity or harsh language",
        
        "education": "\n\nSTYLE: Clear & Educational\n- Break down complex concepts simply\n- Use examples and comparisons\n- Professional, informative tone\n- Focus on learning value",
        
        "engagement": "\n\nSTYLE: Conversational & Discussion-Driven\n- Pose thought-provoking questions\n- Encourage community participation\n- Natural, friendly tone\n- Drive engagement without hype",
        
        "news": "\n\nSTYLE: Journalistic & Factual\n- Present information objectively\n- Include relevant context and implications\n- Neutral, professional language\n- Focus on news value and impact"
    }
    
    base_prompt += style_instructions.get(style, style_instructions["education"])

    if with_hashtags:
        base_prompt += "\n\nHASHTAGS: Include 1-3 relevant, popular hashtags aligned with the content."
    else:
        base_prompt += "\n\nHASHTAGS: Do not include hashtags."

    base_prompt += f"""

FORMATTING RULES:
- Use proper paragraphs (not bullet points unless absolutely necessary)
- Only use bullet points for concise lists (3-5 items max)
- Each bullet point should be ONE SHORT LINE
- Avoid overused phrases: "flipping the script", "epic", "boss", "crew"
- Write naturally, as a human expert would

TOPIC FOCUS: '{topic}'
All content must directly relate to this topic and provide genuine value to the reader.

OUTPUT: English only. Professional presentation with perfect grammar."""

    return base_prompt

@monitor_performance
def call_dobby(system_prompt, user_message, temperature=0.7, is_pro_mode=False, num_outputs=1, chat_history=None):
    max_tokens = 4000 if is_pro_mode else 400
    temperature = min(temperature, 0.9)
    
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    
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
                # Clean up excessive whitespace
                content = re.sub(r'\n{3,}', '\n\n', content)
                content = re.sub(r' {2,}', ' ', content)
                
                # Apply truncation for non-pro mode
                if not is_pro_mode:
                    content = truncate_for_twitter(content)
                
                replies.append(content)
        
        return replies if replies else ["AI service returned empty response."]
    except requests.exceptions.Timeout:
        app.logger.error("Fireworks API timeout")
        return ["Request timeout. Please try again."]
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Fireworks API error: {e}")
        return ["AI service error. Please try again."]
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return ["An unexpected error occurred."]

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
        return jsonify({"error": "FIREWORKS_API_KEY not configured"}), 500
    
    topic = (data.get("message") or "").strip()
    if not topic:
        return jsonify({"error": "Message cannot be empty"}), 400
    if len(topic) > 200:
        return jsonify({"error": "Topic too long (max 200 characters)"}), 400
    
    # Validate and parse parameters
    style = data.get("style", "education")
    if style not in ["meme", "education", "engagement", "news"]:
        style = "education"
    
    is_pro_mode = bool(data.get("proMode", False))
    with_hashtags = bool(data.get("hashtags", True))
    
    try:
        temperature = float(data.get("temperature", 0.7))
        temperature = max(0.1, min(1.5, temperature))
    except (ValueError, TypeError):
        temperature = 0.7
    
    try:
        num_outputs = int(data.get("numOutputs", 1))
        num_outputs = max(1, min(num_outputs, 3))
    except (ValueError, TypeError):
        num_outputs = 1

    # Gather context data
    trending = get_trending_crypto()
    research_context = research_topic(topic)
    twitter_context = get_twitter_sentiment(topic)
    chat_history = session.get("chat_history", [])
    
    # Build system prompt
    system_prompt = build_system_prompt(style, is_pro_mode, with_hashtags, topic)
    
    # Prepare trending text
    trending_text = ""
    if trending:
        trending_names = ", ".join([f"{t['name']} ({t['symbol'].upper()})" for t in trending[:3]])
        trending_text = f"Currently trending: {trending_names}"
    
    # Build user message
    user_message = f"""Analyze and create content about: {topic}

CONTEXT DATA:
Trending: {trending_text}
Twitter Sentiment: {twitter_context}
Research: {research_context}

Style: {style}
Pro Mode: {'Yes' if is_pro_mode else 'No'}
Include Hashtags: {'Yes' if with_hashtags else 'No'}

Please create high-quality, professional content following all guidelines."""
    
    # Generate responses
    replies = call_dobby(system_prompt, user_message, temperature, is_pro_mode, num_outputs, chat_history)
    
    # Filter low-quality responses
    quality_replies = [
        reply for reply in replies
        if len(reply.strip()) > 10 and
        not reply.lower().startswith("i'm sorry") and
        "unavailable" not in reply.lower()
    ]
    
    if not quality_replies:
        quality_replies = ["Unable to generate quality content. Please try adjusting your topic or parameters."]
    
    # Update history
    if quality_replies:
        update_chat_history(topic, quality_replies[0])

    return jsonify({
        "replies": quality_replies,
        "trending": trending,
        "ts": int(time.time())
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": int(time.time())}), 200

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded. Try again later."}), 429

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
else:
    application = app
