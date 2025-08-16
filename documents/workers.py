import io
import json
import os
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
from django.conf import settings


def preprocess_thumbnail_for_boxes(
    image: Image.Image,
) -> List[Tuple[int, int, int, int]]:
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    merged = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = [cv2.boundingRect(c) for c in cnts]
    # convert to (x1,y1,x2,y2)
    return [(x, y, x + w, y + h) for (x, y, w, h) in boxes]


def build_prompt(
    elements_boxes: List[Tuple[int, int, int, int]], excalidraw_data: Dict[str, Any]
) -> str:
    parts = [
        "You are given a whiteboard created with Excalidraw.",
        "Return a concise JSON with: 'summary' (textual description), and 'items' (array).",
        "Each item: {id, type, text?, bbox:[x1,y1,x2,y2]}.",
        "Use the provided bounding boxes and Excalidraw elements to anchor positions.",
    ]
    # Include element ids/types/texts (but not large geometry) for grounding
    elements_brief = []
    for el in (excalidraw_data or {}).get("elements", [])[:200]:
        brief = {
            "id": el.get("id"),
            "type": el.get("type"),
        }
        if "text" in el:
            brief["text"] = el.get("text")
        elements_brief.append(brief)
    parts.append("Excalidraw elements (brief):\n" + json.dumps(elements_brief)[:50000])
    parts.append("Detected bounding boxes:\n" + json.dumps(elements_boxes))
    parts.append("Respond with ONLY the JSON, no prose.")
    return "\n\n".join(parts)


def call_gpt(prompt: str) -> Dict[str, Any]:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("No OpenAI API key found")
        return {"summary": "", "items": []}
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=10000,
        # temperature=0.2,
    )
    try:
        content = resp.output_text
        print(content)
        return json.loads(content)
    except Exception:
        return {"summary": content if "content" in locals() else "", "items": []}


def run_analysis_pipeline(document) -> None:
    if not document.thumbnail:
        # Best effort using boxes from elements frame positions if available
        detected_boxes = []
    else:
        with document.thumbnail.open("rb") as f:
            image = Image.open(f).convert("RGB")
            detected_boxes = preprocess_thumbnail_for_boxes(image)

    prompt = build_prompt(detected_boxes, document.data or {})
    print(prompt)
    result = call_gpt(prompt)
    document.analysis = result
    document.save(update_fields=["analysis", "updated_at"])
