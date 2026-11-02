"""Thin client for the Meta Model API (OpenAI-compatible) and SAM 3.1.

Endpoints and formats follow the public docs at dev.meta.ai:
  * Chat Completions + tool calling : https://api.meta.ai/v1/chat/completions
  * Responses API (image / PDF input, structured output via text.format, SAM 3.1): https://api.meta.ai/v1/responses
Auth is a bearer token from MODEL_API_KEY.
"""
from __future__ import annotations

import base64
import io
import json
import re
import ssl
import time

import httpx
from PIL import Image, ImageOps

from . import config

# Verify TLS against the OS trust store, not httpx's bundled certifi list, so the SAM calls work behind
# antivirus / corporate HTTPS inspection (e.g. Avast) the same way the OpenAI SDK calls already do.
try:
    import truststore

    TLS_VERIFY = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except ImportError:
    TLS_VERIFY = True

# One SAM record is a box optionally followed by a mask:  <0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;...|>
SAM_RECORD = re.compile(r"<\|box;(?P<box>.*?)\|>(?:\s*<\|mask;(?P<mask>.*?)\|>)?", re.S)


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def prepare_image(raw: bytes) -> tuple[bytes, int, int]:
    """Normalize any image to an upright JPEG no larger than MAX_IMAGE_SIDE. Returns (jpeg, width, height)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue(), img.width, img.height


def loads_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("The model did not return valid JSON")


def parse_sam_output(text: str) -> list[dict]:
    """Turn SAM's text output into [{index, box:[x1,y1,x2,y2], w, h, confidence, has_mask}].

    Boxes are in the pixels of the image that was sent. Masks are not decoded here (see README).
    """
    out = []
    for i, m in enumerate(SAM_RECORD.finditer(text or "")):
        raw = [x.strip() for x in m.group("box").split(";") if x.strip()]
        if any("=" in x for x in raw):  # tolerate key=value form
            fields = dict(x.split("=", 1) for x in raw if "=" in x)
        else:  # documented positional form: x1;y1;x2;y2;w;h[;confidence]
            fields = dict(zip(("x1", "y1", "x2", "y2", "w", "h", "c"), raw))
        try:
            box = [int(float(fields[k])) for k in ("x1", "y1", "x2", "y2")]
        except (KeyError, ValueError):
            continue
        out.append(
            {
                "index": i,
                "box": box,
                "w": int(float(fields["w"])) if "w" in fields else None,
                "h": int(float(fields["h"])) if "h" in fields else None,
                "confidence": float(fields["c"]) if "c" in fields else None,
                "has_mask": bool(m.group("mask")),
            }
        )
    return out


class MetaClient:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, max_retries=3, timeout=120)

    # -- Chat Completions with tools (text only: the agent loop never mixes images with tools) -------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs = {"model": config.SPARK_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools  # tool_choice may only be "auto", so we do not set it
        msg = self.client.chat.completions.create(**kwargs).choices[0].message
        out = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ]
        return out

    # -- Responses API: images / PDFs in, schema-constrained JSON out -----------------------------
    def vision_json(self, prompt: str, schema: dict, name: str, images=(), files=()) -> dict:
        content = [{"type": "input_text", "text": prompt}]
        for jpeg in images:
            content.append({"type": "input_image", "image_url": data_url(jpeg, "image/jpeg")})
        for filename, raw, mime in files:
            content.append({"type": "input_file", "filename": filename, "file_data": data_url(raw, mime)})
        resp = self.client.responses.create(
            model=config.SPARK_MODEL,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": False}},
        )
        return loads_json(resp.output_text)

    # -- SAM 3.1: short noun phrase in, boxes (and masks) out ---------------------------------------
    def segment(self, jpeg: bytes, concept: str) -> list[dict]:
        payload = {
            "model": config.SAM_MODEL,
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_image", "image_url": data_url(jpeg, "image/jpeg")},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true"},
        }
        return parse_sam_output(self._sam_request(payload))

    def _sam_request(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        last_error = "unknown error"
        for attempt in range(3):
            try:
                with httpx.stream("POST", f"{config.BASE_URL}/responses", headers=headers, json=payload, timeout=120, verify=TLS_VERIFY) as r:
                    if r.status_code == 400 and "metadata" in payload:
                        payload = {k: v for k, v in payload.items() if k != "metadata"}  # confidence is optional
                        continue
                    if r.status_code in (429, 500, 502, 503, 504):
                        last_error = f"HTTP {r.status_code}"
                        time.sleep(2**attempt)
                        continue
                    r.raise_for_status()
                    text = ""
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        event = json.loads(data)
                        if event.get("type") == "response.output_text.delta":
                            text += event.get("delta", "")
                    return text
            except httpx.TransportError as e:
                last_error = type(e).__name__
                time.sleep(2**attempt)
        raise RuntimeError(f"SAM request failed ({last_error})")


_client = None


def get_client():
    global _client
    if _client is None:
        if config.MOCK_MODE:
            from .mock import MockMeta

            _client = MockMeta()
        else:
            _client = MetaClient()
    return _client
