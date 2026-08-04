import os
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def home():
    return "Homework AI Bot is running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    result = model.generate_content(
        "You are a helpful homework tutor. Explain step by step:\n" + question
    )

    return jsonify({
        "answer": result.text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
