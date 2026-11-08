import os
import re
import time
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
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

# Keep each person's messages in arrival order while allowing different people
# to be handled concurrently.
_message_queues = {}
_active_senders = set()
_message_queue_lock = threading.Lock()
_recent_webhook_events = set()


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
                    is_image BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_image BOOLEAN DEFAULT FALSE;
                CREATE INDEX IF NOT EXISTS idx_user_id_created
                ON messages(user_id, created_at);

                CREATE TABLE IF NOT EXISTS processed_webhook_events (
                    event_id VARCHAR(512) PRIMARY KEY,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_processed_webhook_created
                ON processed_webhook_events(created_at);
            """)
            conn.commit()
            print("Neon Database initialized successfully.")
    except Exception as e:
        print("Database initialization error:", repr(e))
    finally:
        conn.close()

init_db()


def claim_webhook_event(event_id):
    """Atomically claim a Meta message/event key so webhook retries cannot reply twice."""
    if not event_id:
        return True

    conn = get_db_connection()
    if not conn:
        # Without persistent storage, avoid duplicate work within this process.
        with _message_queue_lock:
            if event_id in _recent_webhook_events:
                return False
            _recent_webhook_events.add(event_id)
            return True

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO processed_webhook_events (event_id)
                VALUES (%s)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING event_id;
            """, (event_id,))
            claimed = cur.fetchone() is not None
            cur.execute("""
                DELETE FROM processed_webhook_events
                WHERE created_at < NOW() - INTERVAL '7 days';
            """)
            conn.commit()
            return claimed
    except Exception as e:
        conn.rollback()
        print("Webhook deduplication error:", repr(e))
        # Fail closed: don't send a duplicate reply if deduplication storage fails.
        return False
    finally:
        conn.close()


def cleanup_old_messages(user_id):
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # 1-hour expiration for image interactions
            cur.execute("""
                DELETE FROM messages 
                WHERE user_id = %s 
                AND is_image = TRUE 
                AND created_at < NOW() - INTERVAL '1 hour';
            """, (user_id,))
            
            # 24-hour expiration for standard text interactions
            cur.execute("""
                DELETE FROM messages 
                WHERE user_id = %s 
                AND is_image = FALSE 
                AND created_at < NOW() - INTERVAL '24 hours';
            """, (user_id,))
            conn.commit()
    except Exception as e:
        print("Cleanup DB error:", repr(e))
    finally:
        conn.close()


def save_message(user_id, role, content, is_image=False):
    conn = get_db_connection()
    if not conn:
        return
    try:
        cleanup_old_messages(user_id)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (user_id, role, content, is_image)
                VALUES (%s, %s, %s, %s);
            """, (user_id, role, content, is_image))
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
                SELECT role, content
                FROM (
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE user_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                ) AS recent_messages
                ORDER BY created_at ASC, id ASC;
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
    
    # Strip LaTeX formatting (\mathbf, \text, dollar signs, backslashes)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\([a-zA-Z]+)', r'', text)
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'[\{\}]', '', text)
    
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
    return (
        "You are NOVA, a friendly, capable, general-purpose AI assistant designed to help with ALL subjects and everyday conversations. "
        "NOVA is your only name. Never call yourself EduBot or give NOVA a full form. "
        "Help with every school subject and level when appropriate, including mathematics, science, languages, history, geography, "
        "technology, coding, arts, and general knowledge, as well as cooking, creativity, writing, translation, and everyday questions. "
        "Do not assume every message is homework. Understand the actual request and answer that topic directly. "
        "LANGUAGE RULES ARE STRICT: Identify the language and script of the user's latest actual message, then reply consistently in that same language and script. "
        "English input means English output only. Malayalam-script input means Malayalam-script output only. "
        "Malayalam written with English/Roman letters means Manglish output using English/Roman letters only. "
        "If the user naturally mixes two languages, mirror only those languages in a similar proportion. "
        "Never insert Hindi, Urdu, Arabic, or any other language unless the user used it or explicitly asks for it. "
        "Do not switch scripts mid-sentence. Do not add random translated phrases, multilingual greetings, or language demonstrations. "
        "For very short messages, use the recent conversation to identify the user's language; otherwise use the language of the short message itself. "
        "Answer the actual request first. Do not append generic greetings, homework suggestions, or 'how can I help' questions unless relevant. "
        "Treat the chat as one natural, continuous conversation, like a helpful person chatting normally. "
        "Understand short follow-up messages such as 'yes', 'no', 'para', 'that one', colours, names, and corrections using the recent conversation. "
        "Do not treat each message as an unrelated new conversation, repeat introductions, or ask the user to repeat details already provided. "
        "Respond directly to the latest message while keeping relevant earlier context. Never ignore a message just because it is short or incomplete; ask one brief follow-up only when the context truly cannot resolve it. "
        "Keep replies natural and concise. Do not repeat the same greeting, conclusion, or point in different words. Explain each idea once; do not send three versions of the same answer unless the user asks for alternatives. "
        "When the user sends supported image, audio, video, or document attachments, inspect their contents when the model supports that format. "
        "For stickers, reactions, quick replies, shared posts, story replies, and button taps, use the event context naturally. "
        "If an attachment cannot be accessed or understood, say so briefly and ask for a resend in a supported format; never pretend to have viewed or heard it. "
        "For images, inspect the image and answer the user's actual question about its contents; do not assume it contains mathematics. "
        "If an image arrives without a question, briefly describe what is visible and ask what help is needed. "
        "For homework, explain clearly and show useful steps instead of only giving an unexplained answer. "
        "Be accurate, friendly, and concise. Admit uncertainty and never invent facts. "
        "Use clear plain text suitable for Instagram messages. Avoid LaTeX, backslashes, and excessive markdown."
    )


def ask_gemini(user_id, user_message, edubot_mode=False, media_items=None):
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    system_prompt = build_system_prompt(edubot_mode)
    
    parts = []
    for media in media_items or []:
        media_url = media.get("url")
        if not media_url:
            continue
        media_response = requests.get(media_url, timeout=25)
        media_response.raise_for_status()
        mime_type = (
            media_response.headers.get("Content-Type", "").split(";", 1)[0].strip()
            or media.get("mime_type")
            or "application/octet-stream"
        )
        parts.append(types.Part.from_bytes(data=media_response.content, mime_type=mime_type))

    prompt_text = user_message or "Please inspect the attached content and respond helpfully."
    parts.append(types.Part.from_text(text=prompt_text))

    contents = []
    # Keep the recent chat context for text and image messages alike.
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
            max_output_tokens=2048,
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
        max_tokens=1500
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty response")

    return sanitize_ai_output(answer)


def ask_ai(user_id, user_message, edubot_mode=False, media_items=None):
    media_items = media_items or []
    try:
        reply = ask_gemini(user_id, user_message, edubot_mode, media_items=media_items)
    except Exception as gemini_error:
        print("GEMINI ERROR - switching to Groq:", repr(gemini_error))
        if media_items:
            raise RuntimeError(
                "I received your attachment, but couldn't read it right now. Please try again or send it in another supported format."
            )
        try:
            reply = ask_groq(user_id, user_message, edubot_mode)
        except Exception as groq_error:
            print("GROQ FALLBACK ERROR:", repr(groq_error))
            raise RuntimeError("Both Gemini and Groq are currently unavailable.")

    user_content = user_message or "[Sent an attachment]"
    if media_items:
        labels = ", ".join(item.get("label", item.get("type", "attachment")) for item in media_items)
        user_content = (user_content + " " if user_message else "") + "[Attachments: " + labels + "]"
    save_message(user_id, "user", user_content, is_image=bool(media_items))
    save_message(user_id, "assistant", reply, is_image=bool(media_items))

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

    chunks = [message_text[i:i + 1800] for i in range(0, len(message_text), 1800)]
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    for chunk in chunks:
        payload = {"recipient": {"id": recipient_id}, "message": {"text": chunk}}
        try:
            requests.post(url, json=payload, headers=headers, timeout=15)
            time.sleep(1)  # 1-second delay between sequential message chunks
        except requests.RequestException as e:
            print("Instagram request error:", e)


def process_instagram_message(sender_id, user_message, media_items=None):
    try:
        original_message = (user_message or "").strip()
        lower_message = original_message.lower()
        media_items = media_items or []

        # Image generation remains an explicit user command.
        if not media_items and (lower_message.startswith("image /") or lower_message.startswith("generate image:")):
            prompt = original_message.split('/', 1)[-1].strip() if '/' in original_message else original_message.split(':', 1)[-1].strip()
            if not prompt:
                send_instagram_message(sender_id, "What would you like me to create?")
                return
            gen_url = generate_image_url(prompt)
            send_instagram_media(sender_id, "image", gen_url)
            return

        # Keep NOVA as the only assistant identity; the prefix is optional.
        if lower_message.startswith("nova /"):
            original_message = original_message.split("/", 1)[1].strip()
        elif lower_message.startswith("edubot /"):
            original_message = original_message.split("/", 1)[1].strip()

        if not original_message and not media_items:
            send_instagram_message(sender_id, "I received your message. What would you like help with?")
            return

        reply = ask_ai(sender_id, original_message, edubot_mode=False, media_items=media_items)
        send_instagram_message(sender_id, reply)

    except Exception as e:
        print("BACKGROUND MESSAGE ERROR:", repr(e))
        send_instagram_message(sender_id, "Sorry, I couldn't process that message just now. Please try again.")


def _process_sender_queue(sender_id):
    """Process one sender's queued events sequentially in arrival order."""
    while True:
        with _message_queue_lock:
            queue = _message_queues.get(sender_id)
            if not queue:
                _message_queues.pop(sender_id, None)
                _active_senders.discard(sender_id)
                return
            user_message, media_items = queue.popleft()

        process_instagram_message(sender_id, user_message, media_items)


def enqueue_instagram_message(sender_id, user_message, media_items=None):
    """Queue messages per sender so short follow-ups cannot overtake earlier messages."""
    with _message_queue_lock:
        queue = _message_queues.setdefault(sender_id, deque())
        queue.append((user_message, media_items or []))
        if sender_id not in _active_senders:
            _active_senders.add(sender_id)
            executor.submit(_process_sender_queue, sender_id)


def normalize_instagram_message(message_event):
    """Convert common Instagram message formats into readable context plus media."""
    message = message_event.get("message") or {}
    text_parts = []
    media_items = []

    text_value = (message.get("text") or "").strip()
    if text_value:
        text_parts.append(text_value)

    quick_reply = message.get("quick_reply") or {}
    if quick_reply.get("payload"):
        text_parts.append("[Quick reply: " + str(quick_reply["payload"]) + "]")

    reply_to = message.get("reply_to") or {}
    if reply_to.get("story"):
        text_parts.append("[Reply to a story]")
    elif reply_to.get("mid"):
        text_parts.append("[Reply to an earlier message]")

    for attachment in message.get("attachments", []):
        attachment_type = str(attachment.get("type") or "attachment").lower()
        payload = attachment.get("payload") or {}
        media_url = payload.get("url")
        label = attachment_type.replace("_", " ").title()

        if attachment_type in ("image", "audio", "video", "file") and media_url:
            media_items.append({
                "url": media_url,
                "type": attachment_type,
                "label": label,
            })
            text_parts.append("[Attached " + label.lower() + "]")
        elif attachment_type == "share":
            text_parts.append("[Shared a post or link]")
            if payload.get("url"):
                text_parts.append(str(payload["url"]))
        elif attachment_type in ("story_mention", "ig_reel"):
            text_parts.append("[Shared " + label.lower() + "]")
            if media_url:
                media_items.append({"url": media_url, "type": attachment_type, "label": label})
        else:
            text_parts.append("[Received " + label.lower() + "; attachment details may not be readable]")

    sticker_id = message.get("sticker_id")
    if sticker_id:
        text_parts.append("[Sent a sticker]")

    if message.get("is_unsupported"):
        text_parts.append("[Instagram marked this message as unsupported]")

    return " ".join(part for part in text_parts if part).strip(), media_items


def normalize_instagram_event(message_event):
    """Normalize supported inbound message and postback event shapes."""
    if message_event.get("message"):
        return normalize_instagram_message(message_event)

    postback = message_event.get("postback") or {}
    if postback:
        title = postback.get("title") or "Button"
        payload = postback.get("payload") or ""
        return "[Tapped Instagram button: " + str(title) + (", " + str(payload) if payload else "") + "]", []

    reaction = message_event.get("reaction") or {}
    if reaction:
        emoji = reaction.get("emoji") or reaction.get("action") or "reaction"
        return "[Reacted to a message: " + str(emoji) + "]", []

    return "", []


# -----------------------------
# Base & Legal Endpoints
# -----------------------------
@app.route("/")
def home():
    return "NOVA Multimodal AI is running successfully!"


@app.route("/privacy-policy")
def privacy_policy():
    return "<h1>NOVA Privacy Policy</h1><p>NOVA uses messaging data to provide AI assistance across subjects and general conversations.</p>"


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
        entries = data.get("entry", [])
        # Process every entry and every inbound messaging event in the payload.
        for entry in entries:
            for message_event in entry.get("messaging", []):
                if message_event.get("message", {}).get("is_echo"):
                    continue

                sender_id = message_event.get("sender", {}).get("id")
                if not sender_id:
                    continue

                event_message = message_event.get("message") or {}
                event_id = (
                    event_message.get("mid")
                    or message_event.get("postback", {}).get("mid")
                    or message_event.get("reaction", {}).get("mid")
                )

                # Some event types or payload variants may omit a message ID.
                # Build a stable fallback from the sender, Meta timestamp and
                # message payload so retries of that same delivery are deduped.
                if not event_id:
                    event_timestamp = message_event.get("timestamp")
                    if event_timestamp is not None:
                        attachment_signature = [
                            (
                                item.get("type"),
                                (item.get("payload") or {}).get("url"),
                            )
                            for item in event_message.get("attachments", [])
                        ]
                        fallback_data = repr((
                            sender_id,
                            event_timestamp,
                            event_message.get("text", ""),
                            attachment_signature,
                            message_event.get("postback", {}),
                            message_event.get("reaction", {}),
                        ))
                        event_id = "fallback:" + hashlib.sha256(
                            fallback_data.encode("utf-8")
                        ).hexdigest()

                # Meta can retry webhook deliveries. Only the first delivery
                # of a given message/event key should create an AI reply.
                if event_id and not claim_webhook_event(str(event_id)):
                    print("Duplicate Instagram webhook event ignored:", event_id)
                    continue

                user_message, media_items = normalize_instagram_event(message_event)
                if user_message or media_items:
                    enqueue_instagram_message(sender_id, user_message, media_items)

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
        if lower_question.startswith("nova /") or lower_question.startswith("edubot /"):
            question_for_ai = question.split("/", 1)[1].strip()
            if not question_for_ai:
                return jsonify({"success": False, "error": "Please write your question after NOVA /"}), 400
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
