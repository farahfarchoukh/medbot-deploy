import os
import logging
import requests
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))

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
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        if r.status_code in (401, 403):
            return "Authentication failed with Hugging Face Inference API. Check HF_API_TOKEN (READ scope) and try again."
        if r.status_code == 404:
            return f"Model not found: {HF_MODEL_ID}. Set a valid public model in HF_MODEL_ID."
        return f"Upstream error from Hugging Face: {e}"

    data = r.json()
    # Typical Inference API forms
    if isinstance(data, list) and data and "generated_text" in data[0]:
        text = data[0]["generated_text"]
        return text.split("Assistant:", 1)[-1].strip() + f"\n\n{DISCLAIMER}"
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip() + f"\n\n{DISCLAIMER}"
    if isinstance(data, dict) and "error" in data:
        return f"Hugging Face returned: {data['error']}"
    return "The model did not return a standard response."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    return jsonify(status="ok", model=HF_MODEL_ID, has_token=bool(HF_API_TOKEN))

@app.route("/predict", methods=["POST"])
def predict_form():
    q = (request.form.get("question") or "").strip()
    if not q:
        return render_template("index.html", question=q, answer="Please enter a question.")
    try:
        answer = call_hf_inference(q)
    except Exception as e:
        app.logger.exception("HF API error")
        answer = f"Error: {e}"
    return render_template("index.html", question=q, answer=answer)

if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

