import os
import openai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = openai.OpenAI(
    base_url = "https://api.fireworks.ai/inference/v1",
    api_key = os.getenv("FIREWORKS_API_KEY"),
)

SYSTEM_PROMPT = """
You are Dobby, a brutally honest, cynical, and sarcastic startup coach. 
Your goal is to "roast" a user's startup idea. Be ruthless. Find flaws in the business model, 
the target market, the name, everything. Point out how it's unoriginal, likely to fail, or just plain silly. 
Use sarcasm and humor. Don't hold back. End your roast with one single, small piece of genuinely useful advice, 
almost as an afterthought. Never be positive or encouraging in the main body of your text.
Start your response with "Alright, let's see this brilliant idea of yours..."
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roast', methods=['POST'])
def handle_roast():
    try:
        idea_text = request.json.get('idea')

        if not idea_text:
            return jsonify({'error': 'No idea provided'}), 400
        
        print(f"Menerima ide: {idea_text[:30]}...")
        
        chat_completion = client.chat.completions.create(
            model="accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": idea_text}
            ],
            temperature=0.8,
            max_tokens=512,
        )
        
        roast_result = chat_completion.choices[0].message.content
        print("Roast berhasil dibuat!")

        return jsonify({'roast': roast_result})

    except Exception as e:
        print(f"Terjadi Error: {e}")
        return jsonify({'error': 'Maaf, terjadi kesalahan di sisi server.'}), 500

if __name__ == '__main__':
    app.run(debug=True)