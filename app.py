import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# -----------------------------
# Groq Configuration
# -----------------------------
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

VERIFY_TOKEN = "edubot_verify"


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return "EduBot is running successfully!"


# -----------------------------
# Privacy Policy
# -----------------------------
@app.route("/privacy-policy")
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EduBot Privacy Policy</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>

    <body>
        <h1>EduBot Privacy Policy</h1>

        <p>
        EduBot respects your privacy and is committed to protecting
        your personal information.
        </p>

        <h2>Information We Use</h2>
        <p>
        EduBot uses Instagram messaging data only to provide
        AI-powered homework assistance and replies.
        </p>

        <h2>How We Use Information</h2>
        <p>
        Messages received by EduBot are used only to generate
        helpful AI responses.
        </p>

        <h2>Data Sharing</h2>
        <p>
        We do not sell, rent, or share your personal information
        with third parties.
        </p>

        <h2>Data Security</h2>
        <p>
        We take reasonable steps to protect the information handled
        by EduBot.
        </p>

        <h2>Contact</h2>
        <p>
        For questions about this Privacy Policy, contact:
        your-email@example.com
        </p>

    </body>
    </html>
    """


# -----------------------------
# Meta Webhook Verification
# -----------------------------
@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403


# -----------------------------
# Receive Instagram Messages
# -----------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()

        print("========== WEBHOOK RECEIVED ==========")
        print(data)
        print("======================================")

        return "EVENT_RECEIVED", 200

    except Exception as e:
        print(e)
        return "ERROR", 500


# -----------------------------
# AI Endpoint
# -----------------------------
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
                        "You are EduBot, an AI homework tutor. "
                        "Explain answers step by step using simple language."
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

        return jsonify({
            "success": True,
            "answer": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------
# Instagram Business Login Callback
# -----------------------------
@app.route("/auth/instagram/callback")
def instagram_callback():
    return "Instagram login successful"


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
