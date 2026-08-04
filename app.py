import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize Groq client
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

@app.route("/")
def home():
    return "EduBot is running successfully!"

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
                        "You are EduBot, a friendly AI homework tutor. "
                        "Explain answers step by step in simple language. "
                        "If you don't know something, say so honestly."
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
