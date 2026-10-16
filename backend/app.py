"""
Deepfake Inspector API.

Flask backend for detecting manipulated media:
  * image deepfake detection     (ViT fine-tuned for real vs. deepfake)
  * audio deepfake detection     (wav2vec2 fine-tuned for real vs. fake speech)
  * general image classification (google/vit-base-patch16-224)

Models run locally through Hugging Face transformers by default. Set
INFERENCE_MODE=api and HF_API_TOKEN to use Hugging Face Inference Providers.

Every result includes a plain-English verdict with a confidence score.
"""

from __future__ import annotations

import io
import ipaddress
import logging
import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("deepfake-inspector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", BASE_DIR.parent / "frontend")).resolve()

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").strip().lower()
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
HF_INFERENCE_BASE = os.getenv(
    "HF_INFERENCE_BASE", "https://router.huggingface.co/hf-inference/models"
).rstrip("/")

IMAGE_DEEPFAKE_MODEL = os.getenv("DF_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model")
AUDIO_SPOOF_MODEL = os.getenv(
    "AUDIO_SPOOF_MODEL_ID", "MelodyMachine/Deepfake-audio-detection-V2"
)
IMAGE_CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL_ID", "google/vit-base-patch16-224")

MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 15)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
FETCH_TIMEOUT = _env_int("FETCH_TIMEOUT_SECONDS", 15)
MAX_REDIRECTS = 3
MAX_IMAGE_PIXELS = 40_000_000

IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
AUDIO_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
}

FAKE_WORDS = ("fake", "deepfake", "spoof", "synthetic", "generated", "manipulated", "artificial")
REAL_WORDS = ("real", "realism", "bonafide", "bona fide", "genuine", "authentic", "human")

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.json.sort_keys = False

_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed_origins:
    from flask_cors import CORS

    CORS(app, origins=_allowed_origins)


class UserError(Exception):
    """An error caused by the request. Its message is safe to show the client."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# SSRF-safe URL fetching
# ---------------------------------------------------------------------------

def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _check_url(url: str) -> None:
    """Allow only http(s) URLs whose host resolves exclusively to public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UserError("Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise UserError("URL has no host.")
    if parsed.username or parsed.password:
        raise UserError("URLs with credentials are not allowed.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise UserError("URL has an invalid port.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise UserError("Could not resolve that host.")

    for info in infos:
        if not _is_public_ip(ipaddress.ip_address(info[4][0])):
            raise UserError("That URL points to a private or internal address.", 403)


def fetch_remote_media(url: str) -> tuple[bytes, str, str]:
    """
    Download media from a user-supplied URL with SSRF protections: scheme
    allow-list, private/internal IP block, every redirect re-validated,
    timeout and a hard size cap. Returns (data, content_type, extension).
    """
    url = (url or "").strip()
    if not url:
        raise UserError("No URL provided.")

    headers = {"User-Agent": "DeepfakeInspector/1.0"}
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _check_url(current)
        try:
            resp = requests.get(
                current, headers=headers, timeout=FETCH_TIMEOUT, stream=True, allow_redirects=False
            )
        except requests.RequestException:
            raise UserError("Could not download that URL.", 502)

        if resp.is_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise UserError("Redirect without a destination.", 502)
            current = urljoin(current, location)
            continue

        if resp.status_code != 200:
            resp.close()
            raise UserError(f"Remote server returned HTTP {resp.status_code}.", 502)

        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            resp.close()
            raise UserError(f"Remote file is larger than {MAX_UPLOAD_MB} MB.", 413)

        chunks, total = [], 0
        for chunk in resp.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                resp.close()
                raise UserError(f"Remote file is larger than {MAX_UPLOAD_MB} MB.", 413)
            chunks.append(chunk)

        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        ext = os.path.splitext(urlparse(current).path)[1].lower()
        return b"".join(chunks), content_type, ext

    raise UserError("Too many redirects.", 502)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def read_media_from_request() -> tuple[bytes, str, str]:
    """Return (bytes, content_type, extension) from a multipart upload or JSON {"url": ...}."""
    upload = request.files.get("file")
    if upload and upload.filename:
        data = upload.read()
        if not data:
            raise UserError("Uploaded file is empty.")
        ext = os.path.splitext(upload.filename)[1].lower()
        return data, (upload.mimetype or "").lower(), ext

    payload = request.get_json(silent=True)
    url = payload.get("url") if isinstance(payload, dict) else None
    if not url or not isinstance(url, str):
        raise UserError('Upload a file or send JSON {"url": "..."}.')
    return fetch_remote_media(url)


def media_kind(content_type: str, ext: str) -> str | None:
    if content_type.startswith("image/") or ext in IMAGE_TYPES:
        return "image"
    if content_type.startswith("audio/") or ext in AUDIO_TYPES:
        return "audio"
    return None


def open_image(data: bytes) -> Image.Image:
    """Validate and decode an image. Rejects non-images and decompression bombs."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise UserError("File is not a valid image.")


def image_content_type(raw: bytes, ext: str) -> str:
    if ext in IMAGE_TYPES:
        return IMAGE_TYPES[ext]
    try:
        fmt = (Image.open(io.BytesIO(raw)).format or "").lower()
    except Exception:
        fmt = ""
    return {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "bmp": "image/bmp"}.get(
        fmt, "application/octet-stream"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _local_pipeline(task: str, model_id: str):
    try:
        from transformers import pipeline
    except ImportError:
        raise UserError(
            "Local inference needs transformers and torch (pip install -r requirements.txt), "
            "or set INFERENCE_MODE=api with HF_API_TOKEN.",
            503,
        )
    log.info("Loading %s model %s (the first request can take a minute)", task, model_id)
    return pipeline(task, model=model_id)


def _hf_api(model_id: str, data: bytes, content_type: str) -> list[dict[str, Any]]:
    if not HF_API_TOKEN:
        raise UserError("INFERENCE_MODE=api requires HF_API_TOKEN in the environment.", 503)
    try:
        resp = requests.post(
            f"{HF_INFERENCE_BASE}/{model_id}",
            headers={"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": content_type},
            data=data,
            timeout=120,
        )
    except requests.Timeout:
        raise UserError("Hugging Face inference timed out.", 504)
    except requests.RequestException:
        raise UserError("Could not reach Hugging Face inference.", 502)

    if resp.status_code == 401:
        raise UserError("Hugging Face rejected the token (401).", 502)
    if resp.status_code == 404:
        raise UserError(f"Model '{model_id}' is not served by Hugging Face inference.", 502)
    if resp.status_code == 503:
        raise UserError("Model is loading on Hugging Face. Try again shortly.", 503)
    if resp.status_code != 200:
        raise UserError(f"Hugging Face inference failed (HTTP {resp.status_code}).", 502)

    try:
        result = resp.json()
    except ValueError:
        raise UserError("Hugging Face returned a non-JSON response.", 502)
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        raise UserError("Unexpected response from Hugging Face.", 502)
    return result


def _normalize(preds: Any, top_k: int = 5) -> list[dict[str, Any]]:
    out = [
        {"label": str(p.get("label", "")), "score": round(float(p.get("score", 0.0)), 4)}
        for p in (preds or [])
        if isinstance(p, dict)
    ]
    out.sort(key=lambda p: p["score"], reverse=True)
    return out[:top_k]


def classify_image(model_id: str, img: Image.Image, raw: bytes, ext: str) -> list[dict]:
    if INFERENCE_MODE == "api":
        return _normalize(_hf_api(model_id, raw, image_content_type(raw, ext)))
    return _normalize(_local_pipeline("image-classification", model_id)(img, top_k=5))


def classify_audio(model_id: str, raw: bytes, content_type: str, ext: str) -> list[dict]:
    if INFERENCE_MODE == "api":
        ct = AUDIO_TYPES.get(ext) or content_type or "application/octet-stream"
        return _normalize(_hf_api(model_id, raw, ct))
    classifier = _local_pipeline("audio-classification", model_id)
    try:
        return _normalize(classifier(raw, top_k=5))
    except Exception:
        raise UserError("Could not decode that audio. Install ffmpeg, or try a .wav file.", 422)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _label_side(label: str) -> str | None:
    lower = label.lower().replace("_", " ").replace("-", " ")
    if any(w in lower for w in FAKE_WORDS):
        return "fake"
    if any(w in lower for w in REAL_WORDS):
        return "real"
    return None


def build_verdict(preds: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Turn raw model scores into a plain-English verdict with a confidence level."""
    if not preds:
        return {"verdict": "inconclusive", "confidence": 0.0, "confidence_level": "none",
                "summary": "The model returned no prediction."}

    top = preds[0]
    side = _label_side(top["label"])
    conf = top["score"]
    pct = round(conf * 100, 1)
    noun = "image" if kind == "image" else "audio clip"

    if side is None:
        return {"verdict": "inconclusive", "confidence": conf, "confidence_level": "none",
                "summary": f"The model's top label was '{top['label']}', which doesn't map to real or fake."}

    level = "low" if conf < 0.6 else "medium" if conf < 0.85 else "high"

    if side == "fake":
        what = "likely AI-generated or manipulated" if kind == "image" else "likely synthetic or spoofed"
        advice = "Treat it as untrusted and verify the source before sharing or acting on it."
    else:
        what = "likely authentic"
        advice = "No strong manipulation signal, but detectors can be fooled, so still check the source."

    if level == "low":
        summary = f"Uncertain: this {noun} leans {what} ({pct}% confidence). {advice}"
    else:
        summary = f"This {noun} is {what} ({pct}% confidence). {advice}"

    return {
        "verdict": "likely_fake" if side == "fake" else "likely_real",
        "confidence": conf,
        "confidence_level": level,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "deepfake-inspector", "inference_mode": INFERENCE_MODE})


@app.post("/analyze")
def analyze():
    """Deepfake detection for images, spoof detection for audio."""
    data, content_type, ext = read_media_from_request()
    kind = media_kind(content_type, ext)

    if kind == "image":
        preds = classify_image(IMAGE_DEEPFAKE_MODEL, open_image(data), data, ext)
        model_id = IMAGE_DEEPFAKE_MODEL
    elif kind == "audio":
        preds = classify_audio(AUDIO_SPOOF_MODEL, data, content_type, ext)
        model_id = AUDIO_SPOOF_MODEL
    else:
        raise UserError("Unsupported file type. Use an image or audio file.", 415)

    return jsonify({"type": kind, "model": model_id, **build_verdict(preds, kind), "predictions": preds})


@app.post("/classify-image")
def classify_image_route():
    """General ImageNet classification: what is in the image."""
    data, content_type, ext = read_media_from_request()
    if media_kind(content_type, ext) != "image":
        raise UserError("This endpoint only accepts images.", 415)
    preds = classify_image(IMAGE_CLASSIFIER_MODEL, open_image(data), data, ext)
    return jsonify({
        "type": "image",
        "model": IMAGE_CLASSIFIER_MODEL,
        "predicted_label": preds[0]["label"] if preds else None,
        "predictions": preds,
    })


@app.post("/analyze-image")
def analyze_image_route():
    """Images only: deepfake verdict plus what the image contains."""
    data, content_type, ext = read_media_from_request()
    if media_kind(content_type, ext) != "image":
        raise UserError("This endpoint only accepts images.", 415)
    img = open_image(data)
    df_preds = classify_image(IMAGE_DEEPFAKE_MODEL, img, data, ext)
    cls_preds = classify_image(IMAGE_CLASSIFIER_MODEL, img, data, ext)
    return jsonify({
        "type": "image",
        "model": IMAGE_DEEPFAKE_MODEL,
        **build_verdict(df_preds, "image"),
        "predictions": df_preds,
        "contents": {"model": IMAGE_CLASSIFIER_MODEL, "predictions": cls_preds},
    })


@app.get("/")
def index():
    return app.send_static_file("index.html")


# ---------------------------------------------------------------------------
# Errors and security headers
# ---------------------------------------------------------------------------

@app.errorhandler(UserError)
def handle_user_error(err: UserError):
    return jsonify({"error": err.message}), err.status


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_err):
    return jsonify({"error": f"File is larger than {MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(HTTPException)
def handle_http_error(err: HTTPException):
    return jsonify({"error": err.description}), err.code


@app.errorhandler(Exception)
def handle_unexpected(_err: Exception):
    log.exception("Unhandled error")
    return jsonify({"error": "Internal server error."}), 500


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    )
    return resp


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 5000),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
