import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Groq Client
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# Meta Webhook Verify Token
VERIFY_TOKEN = "edubot_verify"

# Home Route
@app.route("/")
def home():
    return "EduBot is running successfully!"

# Meta Webhook Verification
@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403

# AI Test Route
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()

        if not data or "question" not in data:
            return jsonify({"error": "Question is required"}), 400

        question = data["question"]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are EduBot, a helpful homework tutor. "
                        "Explain answers step by step in simple language."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.5,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
