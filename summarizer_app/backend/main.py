import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Holds the loaded model/tokenizer so we only load them once at startup
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup: load model once ----
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        raise RuntimeError(
            f"No model files found in {MODEL_DIR}. "
            "Copy your saved model folder (config.json, model.safetensors, "
            "tokenizer files, spiece.model, etc.) into backend/model/ first."
        )

    print(f"Loading model from {MODEL_DIR} onto {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    print("Model loaded.")

    yield  # app runs here

    # ---- shutdown: cleanup ----
    ml_models.clear()


app = FastAPI(title="News Summarizer API", lifespan=lifespan)

# Allow the frontend (served separately, e.g. via Live Server / a static host)
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article text to summarize")
    max_length: int = Field(100, ge=10, le=300)
    min_length: int = Field(25, ge=5, le=200)
    num_beams: int = Field(5, ge=1, le=10)


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]

    input_text = "summarize: " + req.text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=req.max_length,
            min_length=req.min_length,
            num_beams=req.num_beams,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return SummarizeResponse(summary=summary)


# Serve the frontend directly from FastAPI too (optional convenience):
# with this, visiting http://localhost:8000/ opens the HTML page,
# and it talks to /summarize on the same origin (no CORS issues).
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
