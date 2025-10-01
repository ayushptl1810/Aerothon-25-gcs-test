import base64
import io
import json
import os
import time
import logging
from typing import List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests


load_dotenv()

app = FastAPI(title="Gemini Frame Analyzer (Legacy)", version="0.4.0")

# Allow local development origins by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ROBOTICS_MODEL = os.getenv("ROBOTICS_MODEL", "models/gemini-robotics-er-1.5-preview")
GEMINI_DEBUG = os.getenv("GEMINI_DEBUG", "").lower() in {"1", "true", "yes"}

# Logging configuration
log_level_name = os.getenv("LOG_LEVEL", "DEBUG" if GEMINI_DEBUG else "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("backend")
logger.info("Backend initialized (legacy): robotics_model=%s, level=%s", ROBOTICS_MODEL, log_level_name)


"""Live/WebRTC related codepaths removed for legacy request-per-frame method."""


def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        # Remove first fence line
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
    if s.endswith("```"):
        s = s[: -3]
    return s.strip()


def _extract_json_array(s: str) -> str:
    """
    Try to extract a JSON array substring from a possibly noisy string (code fences, prose).
    Returns the substring between the first '[' and the matching closing ']' (best-effort).
    """
    if not s:
        return ""
    s = _strip_code_fences(s)
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _encode_jpeg(image_bytes: bytes) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes; cannot decode")
    success, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise ValueError("Failed to encode image as JPEG")
    return bytes(encoded)


def _extract_labels_from_text(text: str) -> List[str]:
    if not text:
        return []
    raw_parts: List[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. ").strip()
        if "," in cleaned:
            raw_parts.extend([p.strip() for p in cleaned.split(",")])
        elif cleaned:
            raw_parts.append(cleaned)
    labels: List[str] = []
    seen = set()
    for part in raw_parts:
        token_count = len(part.split())
        if 0 < token_count <= 6:
            key = part.lower()
            if key not in seen:
                seen.add(key)
                labels.append(part)
    return labels[:10]


# Live SDP endpoint intentionally removed to enforce legacy flow


@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set")
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    try:
        t_start = time.time()
        raw_bytes = await file.read()
        raw_size = len(raw_bytes) if raw_bytes is not None else 0
        logger.debug("/analyze_frame: received bytes=%s", raw_size)
        t_pre0 = time.time()
        jpg_bytes = _encode_jpeg(raw_bytes)
        b64 = base64.b64encode(jpg_bytes).decode("utf-8")

        prompt = (
            "Detect up to 10 distinct objects in the image. Return JSON only as a list of "
            "objects with fields: label (string), point ([y, x]), and bbox ([ymin, xmin, ymax, xmax]). "
            "Coordinates must be normalized to the 0-1000 range. Use bbox when available; "
            "point is the primary anchor (e.g., center or a salient point). Do not add prose. "
            "Example: [{\"label\":\"cup\",\"point\":[400,520],\"bbox\":[320,460,520,640]}]"
        )

        model_path = ROBOTICS_MODEL if ROBOTICS_MODEL.startswith("models/") else f"models/{ROBOTICS_MODEL}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        t_api0 = time.time()
        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=20)
        t_api1 = time.time()
        if resp.status_code != 200:
            logger.warning("Robotics REST non-200: %s %s", resp.status_code, resp.text[:200])
            return {"labels": [], "message": "No labels detected"}

        data = resp.json()
        text = ""
        try:
            cand = data.get("candidates", [])
            if cand:
                parts = cand[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        text += p.get("text", "")
        except Exception:
            text = ""

        labels: List[str] = []
        points: List[List[float]] = []
        bboxes: List[List[float]] = []
        try:
            json_str = _extract_json_array(text)
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for item in parsed:
                    label = item.get("label") if isinstance(item, dict) else None
                    point = item.get("point") if isinstance(item, dict) else None
                    bbox = item.get("bbox") if isinstance(item, dict) else None
                    if isinstance(label, str) and label.strip():
                        labels.append(label.strip())
                    if (
                        isinstance(point, list)
                        and len(point) == 2
                        and all(isinstance(v, (int, float)) for v in point)
                    ):
                        points.append([float(point[0]), float(point[1])])
                    if (
                        isinstance(bbox, list)
                        and len(bbox) == 4
                        and all(isinstance(v, (int, float)) for v in bbox)
                    ):
                        bboxes.append([
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ])
        except Exception as parse_exc:
            logger.debug("/analyze_frame: JSON parse failed, fallback to heuristic labels: %s", parse_exc)
            labels = _extract_labels_from_text(text)

        t_parse1 = time.time()

        if labels:
            total_ms = (t_parse1 - t_start) * 1000.0
            preprocess_ms = (t_api0 - t_pre0) * 1000.0 if 't_pre0' in locals() else 0.0
            api_ms = (t_api1 - t_api0) * 1000.0
            parse_ms = (t_parse1 - t_api1) * 1000.0
            logger.info(
                "/analyze_frame: ok status=%s total_ms=%.1f preprocess_ms=%.1f api_ms=%.1f parse_ms=%.1f labels=%d points=%d bboxes=%d",
                resp.status_code,
                total_ms,
                preprocess_ms,
                api_ms,
                parse_ms,
                len(labels),
                len(points),
                len(bboxes),
            )
            try:
                print("[labels]", ", ".join(labels[:10]))
            except Exception:
                pass
            return {"labels": labels[:10], "points": points[:10], "bboxes": bboxes[:10]}
        total_ms = (t_parse1 - t_start) * 1000.0
        logger.info(
            "/analyze_frame: empty result status=%s total_ms=%.1f", resp.status_code, total_ms
        )
        return {"labels": [], "message": "No labels detected"}

    except Exception as exc:
        logger.exception("/analyze_frame error: %s", exc)
        return JSONResponse(status_code=200, content={"labels": [], "message": "No labels detected"})


