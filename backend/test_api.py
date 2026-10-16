"""
Offline tests for the Deepfake Inspector API.

Model inference and DNS are mocked, so these run in seconds with no network
or model downloads:  pytest -q
"""

import io
import socket

import pytest
from PIL import Image

import app as api


def png_bytes(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    def fake_classify_image(model_id, img, raw, ext):
        if model_id == api.IMAGE_DEEPFAKE_MODEL:
            return [{"label": "Deepfake", "score": 0.93}, {"label": "Realism", "score": 0.07}]
        return [{"label": "tabby cat", "score": 0.8}, {"label": "tiger cat", "score": 0.1}]

    def fake_classify_audio(model_id, raw, content_type, ext):
        return [{"label": "bonafide", "score": 0.71}, {"label": "spoof", "score": 0.29}]

    monkeypatch.setattr(api, "classify_image", fake_classify_image)
    monkeypatch.setattr(api, "classify_audio", fake_classify_audio)
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def upload(client, path, data, filename, mimetype):
    return client.post(path, data={"file": (io.BytesIO(data), filename, mimetype)},
                       content_type="multipart/form-data")


# ---------- Basic routes ----------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_security_headers(client):
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Deepfake Inspector" in r.data


# ---------- Analysis ----------

def test_analyze_image_returns_verdict(client):
    r = upload(client, "/analyze", png_bytes(), "photo.png", "image/png")
    body = r.get_json()
    assert r.status_code == 200
    assert body["type"] == "image"
    assert body["verdict"] == "likely_fake"
    assert body["confidence_level"] == "high"
    assert "93.0% confidence" in body["summary"]


def test_analyze_audio_returns_verdict(client):
    r = upload(client, "/analyze", b"RIFF....WAVEfmt ", "clip.wav", "audio/wav")
    body = r.get_json()
    assert r.status_code == 200
    assert body["type"] == "audio"
    assert body["verdict"] == "likely_real"
    assert body["confidence_level"] == "medium"


def test_full_image_report(client):
    r = upload(client, "/analyze-image", png_bytes(), "photo.png", "image/png")
    body = r.get_json()
    assert r.status_code == 200
    assert body["verdict"] == "likely_fake"
    assert body["contents"]["predictions"][0]["label"] == "tabby cat"


def test_classify_image(client):
    r = upload(client, "/classify-image", png_bytes(), "photo.png", "image/png")
    assert r.status_code == 200
    assert r.get_json()["predicted_label"] == "tabby cat"


# ---------- Input validation ----------

def test_missing_input(client):
    r = client.post("/analyze")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_unsupported_type(client):
    r = upload(client, "/analyze", b"hello", "notes.txt", "text/plain")
    assert r.status_code == 415


def test_fake_image_rejected(client):
    r = upload(client, "/analyze", b"not really a png", "photo.png", "image/png")
    assert r.status_code == 400
    assert "valid image" in r.get_json()["error"]


def test_audio_rejected_on_image_only_endpoint(client):
    r = upload(client, "/classify-image", b"RIFF", "clip.wav", "audio/wav")
    assert r.status_code == 415


def test_upload_size_limit(client):
    big = b"0" * (api.MAX_UPLOAD_BYTES + 1)
    r = upload(client, "/analyze", big, "big.png", "image/png")
    assert r.status_code == 413


def test_errors_do_not_leak_internals(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("secret internal path /Users/x/.env")

    monkeypatch.setattr(api, "classify_image", boom)
    r = upload(client, "/analyze", png_bytes(), "photo.png", "image/png")
    assert r.status_code == 500
    assert "secret" not in r.get_data(as_text=True)


# ---------- SSRF protection ----------

def fake_dns(ip):
    def resolver(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    return resolver


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/a.png",
    "gopher://example.com/",
    "http://user:pass@example.com/a.png",
])
def test_blocks_bad_schemes_and_credentials(client, url):
    r = client.post("/analyze", json={"url": url})
    assert r.status_code == 400


@pytest.mark.parametrize("ip", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private
    "192.168.1.1",      # private
    "169.254.169.254",  # cloud metadata (link-local)
    "0.0.0.0",          # unspecified
])
def test_blocks_internal_addresses(client, monkeypatch, ip):
    monkeypatch.setattr(api.socket, "getaddrinfo", fake_dns(ip))
    r = client.post("/analyze", json={"url": "http://looks-public.example/a.png"})
    assert r.status_code == 403
    assert "private or internal" in r.get_json()["error"]


def test_redirect_to_internal_is_blocked(client, monkeypatch):
    calls = {"n": 0}

    def resolver(host, port, *args, **kwargs):
        ip = "93.184.216.34" if host == "public.example" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    class RedirectResponse:
        is_redirect = True
        status_code = 302
        headers = {"Location": "http://internal.example/admin"}

        def close(self):
            pass

    def fake_get(url, **kwargs):
        calls["n"] += 1
        assert kwargs["allow_redirects"] is False
        return RedirectResponse()

    monkeypatch.setattr(api.socket, "getaddrinfo", resolver)
    monkeypatch.setattr(api.requests, "get", fake_get)
    r = client.post("/analyze", json={"url": "http://public.example/a.png"})
    assert r.status_code == 403
    assert calls["n"] == 1


def test_public_url_is_fetched(client, monkeypatch):
    data = png_bytes()

    class OkResponse:
        is_redirect = False
        status_code = 200
        headers = {"Content-Type": "image/png", "Content-Length": str(len(data))}

        def iter_content(self, _size):
            yield data

        def close(self):
            pass

    monkeypatch.setattr(api.socket, "getaddrinfo", fake_dns("93.184.216.34"))
    monkeypatch.setattr(api.requests, "get", lambda url, **kw: OkResponse())
    r = client.post("/analyze", json={"url": "https://public.example/photo.png"})
    assert r.status_code == 200
    assert r.get_json()["verdict"] == "likely_fake"


# ---------- Verdict logic ----------

@pytest.mark.parametrize("label,score,expected,level", [
    ("Deepfake", 0.95, "likely_fake", "high"),
    ("spoof", 0.7, "likely_fake", "medium"),
    ("Realism", 0.55, "likely_real", "low"),
    ("bonafide", 0.9, "likely_real", "high"),
    ("LABEL_0", 0.9, "inconclusive", "none"),
])
def test_build_verdict(label, score, expected, level):
    v = api.build_verdict([{"label": label, "score": score}], "image")
    assert v["verdict"] == expected
    assert v["confidence_level"] == level


def test_build_verdict_empty():
    assert api.build_verdict([], "audio")["verdict"] == "inconclusive"
