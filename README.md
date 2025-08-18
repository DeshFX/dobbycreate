DobbyCreate
DobbyCreate is a Flask-based web application that leverages AI models to generate professional and engaging social media content, specifically optimized for X (formerly Twitter). It integrates multiple APIs and data sources to enrich the generated content with real-time crypto trends, Twitter sentiment, and contextual research about cryptocurrency topics.

Features
AI-powered content generation using the Fireworks AI model "dobby-unhinged-llama-3-3-70b-new".
Context enrichment from multiple sources:
Trending cryptocurrencies from CoinGecko API.
Recent Twitter sentiment via Twitter V2 API.
Research and search results from Tavily search API.
Rate limiting to prevent abuse (200 requests/hour and 30/min per IP).
Session-based chat history to maintain conversational context.
Support for different styles of tweet content (meme, education, engagement, news).
Two modes of content generation:
Social Media Mode: concise, tweet-length (max 280 characters).
Professional Mode: detailed analysis (500-1000 words).
Automatic handling of tweet length including URL length normalization.
Simple caching for performance optimization.
Health check endpoint and standard error handling.

Installation
Clone the repository:

   
   git clone https://github.com/DeshFX/dobbycreate.git
   cd dobbycreate
   


Create and activate a Python virtual environment (optional but recommended):

   
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   


Install required packages:

   
   pip install -r requirements.txt
   


Create a .env file in the project root with your API keys:

   ```
   FIREWORKS_API_KEY=your_fireworks_ai_api_key
   TAVILY_API_KEY
=your_tavily_api_key
   COINGECKO_API_KEY=your_coingecko_api_key  # optional
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token  # optional
   
## Usage

1. Run the Flask server:

   
bash
   python app.py
   
2. By default, the app will be available at `http://localhost:5000`.

3. The main chat interface is served on the root URL `/`. Interact by submitting cryptocurrency topics or keywords.

4. The backend generates content based on your inputs, enriched with live data and research context.

## API Endpoints

- `GET /` - Returns the chat web interface.
- `POST /chat` - JSON endpoint to send a topic and receive AI-generated content.

  Example request JSON:

  
json
  {
    "message": "bitcoin",
    "style": "education",
    "proMode": false,
    "hashtags": true,
    "temperature": 0.7,
    "numOutputs": 1
  }
  ``

  Response JSON contains AI replies, trending crypto data, and timestamp.

- GET /health - Returns server health status.

## Configuration

- Rate limits per IP: 200 requests per hour, 30 requests per minute.
- Session lifetime: 2 hours.
- AI model and API URL configured for the Fireworks AI platform.
- Twitter client uses Tweepy library with Twitter API V2 bearer token.
- Caching is provided using Flask-Caching simple cache.

## Project Structure

- app.py: Main Flask application, routing, API integrations, caching, and AI content generation.
- templates/: HTML template for the chat interface.
- requirements.txt: Python dependencies.
- .gitignore: Files to ignore in git.
- vercel.json`: Configuration file for Vercel deployment.

Notes and Limitations
Twitter and Tavily API usage requires valid API keys, otherwise those features are disabled.
Limited error handling for external API failures.
Content generation errors fall back to polite error messages.
Designed for cryptocurrency-related content generation but can be ad
apted.
No frontend SPA framework; uses Flask server-rendered templates.
Single-threaded Flask server; consider production WSGI deployment.

License
This project is open source under the MIT License.

---

You can modify or extend the README if you want more specific usage examples or deployment instructions. Let me know if you'd like help with that.
