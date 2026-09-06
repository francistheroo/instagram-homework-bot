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

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) if os.environ.get("GEMINI_API_KEY") else None
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

VERIFY_TOKEN = "edubot_verify"
executor = ThreadPoolExecutor(max_workers=4)


# -----------------------------
# Neon PostgreSQL Setup & Helpers
# -----------------------------
def get_db_connection():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    """Create conversation table and purge old records on startup."""
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

# Initialize DB on start
init_db()


def cleanup_old_messages(user_id):
    """Automatic deletion of messages older than 24 hours."""
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
    """Save a user or assistant message to Neon DB."""
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
    """Fetch recent message history within the last 24 hours."""
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
# Formatting & Sanitization Helpers
# -----------------------------
def sanitize_ai_output(text: str) -> str:
    """Removes token leakage like [ }{46], internal reasoning tags, or prompt markers."""
    if not text:
        return ""
    text = re.sub(r'\[\s*\\?\}{0,2}\d*\s*\]', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^(System|Assistant|User):', '', text, flags=re.IGNORECASE)
    return text.strip()


def clean_markdown_for_instagram(text: str) -> str:
    """Converts raw Markdown tables, headings, and bold syntax into clean plain text for Instagram DM."""
    if not text:
        return ""

    # Convert Markdown Tables (| header |) into plain bullet points
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip table divider rows like |---|---|
        if re.match(r'^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?$', stripped):
            continue
        # Format table rows
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.split('|')[1:-1] if c.strip()]
            if len(cells) >= 2:
                cleaned_lines.append(f"• {cells[0]}: {cells[1]}")
            elif len(cells) == 1:
                cleaned_lines.append(f"• {cells[0]}")
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Convert headers (### Heading -> HEADING)
    text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)

    # Strip bold and italic asterisks/underscores
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # Clean multiple trailing blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# -----------------------------
# AI Prompt & Generation
# -----------------------------
def build_system_prompt(edubot_mode=False):
    if edubot_mode:
        return (
            "You are EduBot, an AI homework tutor. "
            "Explain answers step by step using simple, clear plain text. "
            "Do not use markdown tables. Help the student understand the method."
        )
    return (
        "You are a friendly AI assistant having a normal conversation. "
        "Answer naturally and clearly in plain text. Do not use markdown tables or complex formatting. "
        "Do not behave as a homework tutor unless the user starts the message with 'EduBot /'."
    )


def ask_gemini(user_id, user_message, edubot_mode=False):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    system_prompt = build_system_prompt(edubot_mode)
    history = get_conversation_history(user_id)

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            max_output_tokens=800,
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


def ask_ai(user_id, user_message, edubot_mode=False):
    """Primary: Gemini | Fallback: Groq"""
    try:
        reply = ask_gemini(user_id, user_message, edubot_mode)
    except Exception as gemini_error:
        print("GEMINI ERROR - switching to Groq:", repr(gemini_error))
        try:
            reply = ask_groq(user_id, user_message, edubot_mode)
        except Exception as groq_error:
            print("GROQ FALLBACK ERROR:", repr(groq_error))
            raise RuntimeError("Both Gemini and Groq are currently unavailable.")

    save_message(user_id, "user", user_message)
    save_message(user_id, "assistant", reply)

    return reply


# -----------------------------
# Base & Legal Endpoints
# -----------------------------
@app.route("/")
def home():
    return "EduBot is running successfully!"


@app.route("/privacy-policy")
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>EduBot Privacy Policy</title></head>
    <body>
    <h1>EduBot Privacy Policy</h1>
    <p>EduBot respects your privacy and protects your personal information.</p>
    <h2>Information We Use</h2>
    <p>EduBot uses Instagram messaging data only to provide AI-powered homework assistance and replies.</p>
    <h2>Data Usage</h2>
    <p>Messages are used only to generate helpful AI responses.</p>
    <h2>Data Sharing</h2>
    <p>We do not sell, rent, or share personal information with third parties.</p>
    <h2>Contact</h2>
    <p>Email: zedexeditingz@gmail.com</p>
    </body>
    </html>
    """


@app.route("/data-deletion")
def data_deletion():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>EduBot Data Deletion</title></head>
    <body>
    <h1>User Data Deletion</h1>
    <p>Users can request deletion of their data from EduBot.</p>
    <p>Please contact us with your Instagram username.</p>
    <p>We will review your request and delete applicable data.</p>
    <p>Email: zedexeditingz@gmail.com</p>
    </body>
    </html>
    """


@app.route("/terms")
def terms():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>EduBot Terms of Service</title></head>
    <body>
    <h1>EduBot Terms of Service</h1>
    <p>Welcome to EduBot. By using this service, you agree to these Terms of Service.</p>
    <h2>Educational Purpose</h2>
    <p>EduBot provides AI-powered educational assistance and homework support.</p>
    <h2>AI Responses</h2>
    <p>Responses are generated by artificial intelligence and may not always be accurate. Users should verify important information independently.</p>
    <h2>Acceptable Use</h2>
    <p>Users must not use EduBot for illegal, abusive, harmful, or unauthorized purposes.</p>
    <h2>Privacy</h2>
    <p>Your use of EduBot is also governed by our Privacy Policy.</p>
    <h2>Contact</h2>
    <p>Email: zedexeditingz@gmail.com</p>
    </body>
    </html>
    """


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


def send_instagram_message(recipient_id, message_text):
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    url = "https://graph.instagram.com/v23.0/me/messages"

    # Clean formatting for Instagram DM output
    message_text = clean_markdown_for_instagram(message_text)
    if not message_text:
        message_text = "Sorry, I couldn't generate a response."

    chunks = [message_text[i:i + 1000] for i in range(0, len(message_text), 1000)]
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    for chunk in chunks:
        payload = {"recipient": {"id": recipient_id}, "message": {"text": chunk}}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            if not response.ok:
                print(f"Instagram API error: HTTP {response.status_code}")
        except requests.RequestException as e:
            print("Instagram request error:", e)


def process_instagram_message(sender_id, user_message):
    try:
        original_message = user_message.strip()
        lower_message = original_message.lower()

        if lower_message.startswith("edubot /"):
            question = original_message[len("edubot /"):].strip()
            reply = "Please write your question after EduBot /" if not question else ask_ai(sender_id, question, edubot_mode=True)
        else:
            reply = ask_ai(sender_id, original_message, edubot_mode=False)

        send_instagram_message(sender_id, reply)

    except Exception as e:
        print("BACKGROUND MESSAGE ERROR:", repr(e))
        send_instagram_message(sender_id, "Sorry, I couldn't process that right now. Please try again.")


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

        if sender_id and user_message:
            executor.submit(process_instagram_message, sender_id, user_message)

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
