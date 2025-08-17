import os, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import torch

# -------- Settings --------
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.0
TOP_P = 1.0
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 3

SYSTEM_PROMPT = (
    "You are MedBot, a careful, evidence-based medical assistant.\n"
    "Provide concise, professional, medically accurate information. "
    "Include definition/overview, key causes or risk factors, main complications, "
    "evidence-based management or prevention. Avoid casual tone. "
    "Do *not* diagnose or prescribe. "
    "If it sounds urgent, advise seeking medical attention immediately."
)

SAFETY_FOOTER = (
    "When to seek care: new/worsening symptoms, severe pain, trouble breathing, neurologic symptoms, "
    "signs of emergency, or anything concerning → seek urgent medical attention.\n"
    "⚠️ Disclaimer: This is general information, not a medical diagnosis or treatment plan."
)

def _build_prompt(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return SYSTEM_PROMPT + "\n\nQuestion: (none)\nAnswer:"
    return f"{SYSTEM_PROMPT}\n\nQuestion: {q}\nAnswer:"

def _postprocess(text: str) -> str:
    if not text:
        return ""
    # remove stray role tags
    text = re.sub(r"(?i)\b(User|Assistant|System)\s*:\s*", "", text).strip()
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # enforce footer
    if "When to seek care" not in text:
        text = text.rstrip() + "\n\n" + SAFETY_FOOTER
    return text.strip()

def load_model():
    # CPU-safe dtype and single-thread helps stability on small machines
    torch.set_num_threads(1)
    gen = pipeline(
        task="text-generation",
        model=DEFAULT_MODEL,
        device_map="cpu",
        model_kwargs={"low_cpu_mem_usage": True},
        torch_dtype=torch.float32,
    )
    # warmup (small)
    try:
        gen("Warmup.", max_new_tokens=8, do_sample=False, temperature=0.0, top_p=1.0, return_full_text=False)
    except Exception:
        pass
    return {"pipe": gen, "model_id": DEFAULT_MODEL}

def infer(model, question: str) -> str:
    pipe = model["pipe"]
    prompt = _build_prompt(question)
    try:
        out = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            return_full_text=False,
        )
        text = (out[0].get("generated_text") or "").strip()
        return _postprocess(text) or "Sorry, I couldn’t generate an answer. Please try rephrasing."
    except Exception as e:
        return f"An internal error occurred during generation: {e}\n\n{SAFETY_FOOTER}"
