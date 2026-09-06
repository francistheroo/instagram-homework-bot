import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,
    max_retries=2
)

gemini_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "edubot_verify")
executor = ThreadPoolExecutor(max_workers=4)


# -----------------------------
# Database Setup & Retention
# -----------------------------
def get_db_connection():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    conn = get_db_connection()
    if not conn:
        print("WARNING: DATABASE_URL not provided. Database functionality disabled.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_user_id_created 
                ON messages(user_id, created_at);
            """)
            conn.commit()
            print("Neon Database initialized successfully.")
    except Exception as e:
        print("Database initialization error:", repr(e))
    finally:
        conn.close()

init_db()


def cleanup_old_messages(user_id):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM messages 
                WHERE user_id = %s 
                AND created_at < NOW() - INTERVAL '24 hours';
            """, (user_id,))
            conn.commit()
    except Exception as e:
        print("Cleanup DB error:", repr(e))
    finally:
        conn.close()


def save_message(user_id, role, content):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cleanup_old_messages(user_id)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (user_id, role, content)
                VALUES (%s, %s, %s);
            """, (user_id, role, content))
            conn.commit()
    except Exception as e:
        print("Save message error:", repr(e))
    finally:
        conn.close()


def get_conversation_history(user_id, limit=10):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cleanup_old_messages(user_id)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content FROM messages
                WHERE user_id = %s
                ORDER BY created_at ASC
                LIMIT %s;
            """, (user_id, limit))
            return cur.fetchall()
    except Exception as e:
        print("Get history error:", repr(e))
        return []
    finally:
        conn.close()


# -----------------------------
# Sanitization & Formatting
# -----------------------------
def sanitize_ai_output(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[\s*\\?\}{0,2}\d*\s*\]', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^(System|Assistant|User):', '', text, flags=re.IGNORECASE)
    return text.strip()


def clean_markdown_for_instagram(text: str) -> str:
    if not text:
        return ""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?$', stripped):
            continue
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.split('|')[1:-1] if c.strip()]
            if len(cells) >= 2:
                cleaned_lines.append(f"• {cells[0]}: {cells[1]}")
            elif len(cells) == 1:
                cleaned_lines.append(f"• {cells[0]}")
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# -----------------------------
# AI Prompt & Multimodal Logic
# -----------------------------
def build_system_prompt(edubot_mode=False):
    if edubot_mode:
        return (
            "You are EduBot, an expert AI math and homework tutor. "
            "When given an image, inspect it carefully and solve all visible math problems or questions step-by-step. "
            "Keep explanations simple, clear, and direct without using markdown tables."
        )
    return (
        "You are a friendly AI assistant having a normal conversation. "
        "Answer naturally and clearly in plain text."
    )


def ask_gemini(user_id, user_message, edubot_mode=False, image_url=None):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    system_prompt = build_system_prompt(edubot_mode)
    
    parts = []
    if image_url:
        img_data = requests.get(image_url).content
        parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
        
    prompt_text = user_message or "Solve all the math problems shown in this image step-by-step."
    parts.append(types.Part.from_text(text=prompt_text))

    contents = []
    if not image_url:
        history = get_conversation_history(user_id)
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    contents.append(types.Content(role="user", parts=parts))

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=1000,
        ),
    )

    answer = getattr(response, "text", None)
    if not answer:
        raise RuntimeError("Gemini returned an empty response")

    return sanitize_ai_output(answer)


def ask_groq(user_id, user_message, edubot_mode=False):
    system_prompt = build_system_prompt(edubot_mode)
    history = get_conversation_history(user_id)

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=800
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty response")

    return sanitize_ai_output(answer)


def ask_ai(user_id, user_message, edubot_mode=False, image_url=None):
    try:
        reply = ask_gemini(user_id, user_message, edubot_mode, image_url=image_url)
    except Exception as gemini_error:
        print("GEMINI ERROR - switching to Groq:", repr(gemini_error))
        if image_url:
            raise RuntimeError("Vision capability requires Gemini API.")
        try:
            reply = ask_groq(user_id, user_message, edubot_mode)
        except Exception as groq_error:
            print("GROQ FALLBACK ERROR:", repr(groq_error))
            raise RuntimeError("Both Gemini and Groq are currently unavailable.")

    save_message(user_id, "user", user_message or "[Sent an Image]")
    save_message(user_id, "assistant", reply)

    return reply


# -----------------------------
# Image Generation Helper
# -----------------------------
def generate_image_url(prompt):
    encoded_prompt = urllib.parse.quote(prompt.strip())
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"


# -----------------------------
# Instagram Dispatcher
# -----------------------------
def send_instagram_media(recipient_id, media_type, media_url):
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    url = "https://graph.instagram.com/v23.0/me/messages"

    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": media_type,
                "payload": {"url": media_url}
            }
        }
    }
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    requests.post(url, json=payload, headers=headers, timeout=15)


def send_instagram_message(recipient_id, message_text):
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    url = "https://graph.instagram.com/v23.0/me/messages"

    message_text = clean_markdown_for_instagram(message_text)
    if not message_text:
        message_text = "Sorry, I couldn't generate a response."

    chunks = [message_text[i:i + 1000] for i in range(0, len(message_text), 1000)]
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    for chunk in chunks:
        payload = {"recipient": {"id": recipient_id}, "message": {"text": chunk}}
        try:
            requests.post(url, json=payload, headers=headers, timeout=15)
        except requests.RequestException as e:
            print("Instagram request error:", e)


def process_instagram_message(sender_id, user_message, image_url=None):
    try:
        original_message = (user_message or "").strip()
        lower_message = original_message.lower()

        # 1. Image Generation Command
        if lower_message.startswith("image /") or lower_message.startswith("generate image:"):
            prompt = original_message.split('/', 1)[-1].strip() if '/' in original_message else original_message.split(':', 1)[-1].strip()
            gen_url = generate_image_url(prompt)
            send_instagram_media(sender_id, "image", gen_url)
            return

        # 2. Vision Mode (Received an Image Attachment)
        if image_url:
            prompt = original_message if original_message else "Solve all the math problems shown in this image step-by-step."
            reply = ask_ai(sender_id, prompt, edubot_mode=True, image_url=image_url)
            send_instagram_message(sender_id, reply)
            return

        # 3. EduBot Homework Mode
        if lower_message.startswith("edubot /"):
            question = original_message[len("edubot /"):].strip()
            reply = "Please write your question after EduBot /" if not question else ask_ai(sender_id, question, edubot_mode=True)
            send_instagram_message(sender_id, reply)
            return

        # 4. Standard Text Chat Mode
        reply = ask_ai(sender_id, original_message, edubot_mode=False)
        send_instagram_message(sender_id, reply)

    except Exception as e:
        print("BACKGROUND MESSAGE ERROR:", repr(e))
        send_instagram_message(sender_id, "Sorry, I couldn't process that right now. Please try again.")


# -----------------------------
# Base & Legal Endpoints
# -----------------------------
@app.route("/")
def home():
    return "EduBot Multimodal AI is running successfully!"


@app.route("/privacy-policy")
def privacy_policy():
    return "<h1>EduBot Privacy Policy</h1><p>EduBot uses messaging data only to provide AI homework support.</p>"


@app.route("/data-deletion")
def data_deletion():
    return "<h1>Data Deletion</h1><p>Contact support with your Instagram handle to purge records.</p>"


@app.route("/terms")
def terms():
    return "<h1>Terms of Service</h1><p>Educational AI service provided as-is.</p>"


# -----------------------------
# Webhook & Routes
# -----------------------------
@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or {}
        entry = data.get("entry", [])
        if not entry:
            return "EVENT_RECEIVED", 200

        messaging = entry[0].get("messaging", [])
        if not messaging:
            return "EVENT_RECEIVED", 200

        message_event = messaging[0]
        message = message_event.get("message", {})

        if not message or message.get("is_echo"):
            return "EVENT_RECEIVED", 200

        sender_id = message_event.get("sender", {}).get("id")
        user_message = message.get("text", "").strip()

        image_url = None
        attachments = message.get("attachments", [])
        if attachments and attachments[0].get("type") == "image":
            image_url = attachments[0].get("payload", {}).get("url")

        if sender_id and (user_message or image_url):
            executor.submit(process_instagram_message, sender_id, user_message, image_url)

        return "EVENT_RECEIVED", 200
    except Exception as e:
        print("WEBHOOK ERROR:", repr(e))
        return "EVENT_RECEIVED", 200


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        user_id = str(data.get("user_id", "web_default_user"))

        if not question:
            return jsonify({"error": "Question is required"}), 400

        lower_question = question.lower()
        if lower_question.startswith("edubot /"):
            question_for_ai = question[len("edubot /"):].strip()
            if not question_for_ai:
                return jsonify({"success": False, "error": "Please write your question after EduBot /"}), 400
            answer = ask_ai(user_id, question_for_ai, edubot_mode=True)
        else:
            answer = ask_ai(user_id, question, edubot_mode=False)

        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        print("ASK ERROR:", repr(e))
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
