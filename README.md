# Deepfake Inspector

**Winner, "Preventing Scams and Cyber Threats" track, DevHacks S2 (ASU DevLabs, September 2025)**

Scammers now use AI-generated photos and cloned voices to impersonate real people. Deepfake Inspector lets anyone upload an image or voice clip, or paste a link, and get a plain-English verdict on whether it looks real or synthetic, with a confidence score.

## What it does

| Mode | Input | Output |
|---|---|---|
| Deepfake check | image or audio | Likely fake / likely authentic / inconclusive, with confidence |
| Full image report | image | Deepfake verdict plus what the image shows |
| What's in the image | image | Top ImageNet labels |

Models (all from Hugging Face, Apache-2.0):

- Images: [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) (ViT, Realism vs. Deepfake)
- Audio: [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) (wav2vec2, real vs. fake speech)
- Image contents: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224)

Models run locally by default. Set `INFERENCE_MODE=api` to use Hugging Face Inference Providers instead.

## Security design

The app accepts untrusted files and URLs, so the backend treats every input as hostile:

- **SSRF protection on URL analysis.** Only `http`/`https` is allowed, URLs with embedded credentials are rejected, and the host must resolve only to public IPs (loopback, private ranges, link-local and cloud metadata like `169.254.169.254` are blocked). Redirects are followed manually and every hop is re-checked, so a public URL can't bounce the server into the internal network.
- **Resource limits.** 15 MB cap on uploads and downloads (enforced while streaming, not after), request timeouts, and Pillow's decompression-bomb guard.
- **Content validation.** Images are decoded and verified, not trusted by extension or MIME type.
- **No secrets in code.** Tokens come from environment variables (`.env` is git-ignored).
- **Safe errors.** Clients get short error messages and stack traces stay in server logs.
- **Browser hardening.** Content-Security-Policy, `X-Frame-Options: DENY`, `nosniff`, `no-referrer`. CORS is off unless origins are explicitly allow-listed. The frontend never builds HTML from user or model data.

These protections are covered by tests in `backend/test_api.py`.

## Run it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional; defaults work without it
python app.py
```

Open http://127.0.0.1:5000. The first analysis downloads the models (a few hundred MB), so it takes a minute. Audio formats other than WAV need `ffmpeg` (`brew install ffmpeg`).

## Tests

```bash
cd backend && pytest -q
```

The tests mock model inference and DNS, so they run offline in about a second. They cover the API routes, verdict logic, input validation, upload limits, error handling and SSRF cases (blocked schemes, internal IPs, redirect-to-internal).

## API

| Method | Path | Body |
|---|---|---|
| GET | `/health` | none |
| POST | `/analyze` | multipart `file`, or JSON `{"url": "..."}` |
| POST | `/analyze-image` | same, images only |
| POST | `/classify-image` | same, images only |

Example response from `/analyze`:

```json
{
  "type": "image",
  "model": "prithivMLmods/Deep-Fake-Detector-v2-Model",
  "verdict": "likely_fake",
  "confidence": 0.93,
  "confidence_level": "high",
  "summary": "This image is likely AI-generated or manipulated (93.0% confidence). Treat it as untrusted and verify the source before sharing or acting on it.",
  "predictions": [{"label": "Deepfake", "score": 0.93}, {"label": "Realism", "score": 0.07}]
}
```

## Limitations

Deepfake detectors are trained on specific generators and can miss new ones or flag heavily edited real photos. Treat the verdict as one signal, not proof.

## Team

Built at DevHacks S2 by [@aayouiyer](https://github.com/aayouiyer), [@audhaniAashi](https://github.com/audhaniAashi) and [@jrupani0512](https://github.com/jrupani0512).
