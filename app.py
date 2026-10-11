"""
AI Voice Tutor - Flask backend
Flow: browser records voice -> /api/transcribe (Groq Whisper)
      -> /api/chat (Groq GPT-OSS) -> browser speaks the answer aloud.

Models can be changed in the .env file without touching this code:
    CHAT_MODEL, FALLBACK_MODEL, STT_MODEL
Check current model names at https://console.groq.com/docs/models
"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY (and optional model names) from .env

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Current Groq production models (self-serve, October 2026)
STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")      # speech -> text
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")       # tutor brain
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "openai/gpt-oss-20b")  # used if the first fails
MAX_HISTORY = 10  # keep only the last 10 messages


def build_system_prompt(grade: str, subject: str) -> str:
    """The rules that make the AI behave like a kind school tutor."""
    return f"""You are Tara, a friendly voice tutor for a school student in {grade}.
The student wants help with: {subject}.

Rules:
1. Explain in simple words a {grade} student understands. Use a short everyday example.
2. Your answer will be read aloud, so write plain sentences only.
   No markdown, no bullet points, no emojis, no symbols like * or #.
3. Keep answers to 3-5 short sentences.
4. For homework problems, guide step by step and let the student try the last step.
   Do not just hand over the final answer.
5. End with one small question to check understanding.
6. Be kind and encouraging. If the student is wrong, say so gently and explain why.
7. Only discuss school learning. If asked about anything unsafe or unrelated,
   politely steer back to studying. If the student seems upset or unsafe,
   encourage them to talk to a parent, teacher, or other trusted adult.
8. Reply in the language the student uses."""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Turn the student's recorded voice into text."""
    audio = request.files.get("audio")
    if audio is None:
        return jsonify(error="No audio received."), 400

    try:
        result = client.audio.transcriptions.create(
            file=(audio.filename or "speech.webm", audio.read()),
            model=STT_MODEL,
            response_format="json",
        )
        return jsonify(text=result.text.strip())
    except Exception:
        app.logger.exception("Transcription failed")
        return jsonify(error="Could not understand the audio. Please try again."), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send the conversation to the tutor model and return its answer."""
    data = request.get_json(silent=True) or {}
    history = data.get("messages", [])[-MAX_HISTORY:]
    grade = data.get("grade", "Class 6")
    subject = data.get("subject", "any subject")

    if not history:
        return jsonify(error="Say or type something first."), 400

    # Only allow safe roles/content coming from the browser
    clean = [
        {"role": m["role"], "content": str(m["content"])[:1000]}
        for m in history
        if m.get("role") in ("user", "assistant")
    ]
    messages = [{"role": "system", "content": build_system_prompt(grade, subject)}] + clean

    # Try the main model first, then the fallback (dict.fromkeys removes duplicates)
    for model in dict.fromkeys([CHAT_MODEL, FALLBACK_MODEL]):
        try:
            options = {}
            if "gpt-oss" in model:
                # GPT-OSS models "think" before answering; low effort keeps replies fast
                options["extra_body"] = {"reasoning_effort": "low"}

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_completion_tokens=1024,  # room for thinking + a short answer
                **options,
            )
            reply = (completion.choices[0].message.content or "").strip()
            if reply:
                return jsonify(reply=reply)
            app.logger.warning("Model %s returned an empty reply", model)
        except Exception as exc:
            app.logger.warning("Model %s failed: %s", model, exc)

    app.logger.error("All chat models failed. Check your key and model names.")
    return jsonify(error="The tutor is busy. Please try again in a moment."), 500


if __name__ == "__main__":
    app.run(debug=True)
