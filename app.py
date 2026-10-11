"""
AI Voice Tutor - Flask backend
Flow: browser records voice -> /api/transcribe (Groq Whisper)
      -> /api/chat (Groq Llama) -> browser speaks the answer aloud.
"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from the .env file

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

STT_MODEL = "whisper-large-v3-turbo"   # speech -> text
CHAT_MODEL = "llama-3.3-70b-versatile"  # the tutor's brain
MAX_HISTORY = 10                        # keep only the last 10 messages


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
    except Exception as exc:  # network error, bad key, rate limit...
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

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": build_system_prompt(grade, subject)}]
            + clean,
            temperature=0.5,
            max_tokens=300,
        )
        return jsonify(reply=completion.choices[0].message.content.strip())
    except Exception:
        app.logger.exception("Chat failed")
        return jsonify(error="The tutor is busy. Please try again in a moment."), 500


if __name__ == "__main__":
    app.run(debug=True)
