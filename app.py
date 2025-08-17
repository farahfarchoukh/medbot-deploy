import os
import logging
import requests
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TIMEOUT = float(os.getenv("HF_TIMEOUT", "30"))

SYSTEM_PROMPT = (
    "You are MedBot, an AI medical assistant. Provide concise, clinically sound, "
    "evidence-based answers. Structure: (1) Definition (2) Causes/Risk factors "
    "(3) Key complications (4) Evidence-based self-care/management (5) ‘When to seek care’. "
    "Professional tone. No questions to the user unless essential. Include a brief disclaimer."
)

DISCLAIMER = ("⚠️ Disclaimer: This is general information, not a medical diagnosis or "
              "treatment plan. For emergencies, seek immediate medical care.")

def call_hf_inference(user_question: str) -> str:
    if not HF_API_TOKEN:
        return "Server not configured with HF_API_TOKEN."
    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_question}\nAssistant:"
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 180, "temperature": 0.1}}
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = data[0]["generated_text"]
        return text.split("Assistant:", 1)[-1].strip() + f"\n\n{DISCLAIMER}"
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip() + f"\n\n{DISCLAIMER}"
    return "The model did not return a standard response."

@app.route("/")
def home():
    # expects templates/index.html (you already have one)
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    return jsonify(status="ok")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    question = (request.get_json() or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Missing 'question'"), 400
    try:
        answer = call_hf_inference(question)
        return jsonify(answer=answer)
    except Exception as e:
        app.logger.exception("HF API error")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
