import base64
import io
import json
import os
from typing import Dict, Any, List, Tuple
from PIL import Image
from django.core.files.base import ContentFile
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
    # elements_brief = []
    # for el in (excalidraw_data or {}).get("elements", [])[:200]:
    #     brief = {
    #         "id": el.get("id"),
    #         "type": el.get("type"),
    #     }
    #     if "text" in el:
    #         brief["text"] = el.get("text")
    #     elements_brief.append(brief)
    # parts.append("Excalidraw elements (brief):\n" + json.dumps(elements_brief)[:50000])
    parts.append("Detected bounding boxes:\n" + json.dumps(elements_boxes))
    parts.append("Respond with ONLY the JSON, no prose.")
    return "\n\n".join(parts)


def get_document_analysis(prompt: str, document) -> Dict[str, Any]:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("No OpenAI API key found")
        return {"summary": "", "items": []}
    client = OpenAI(api_key=api_key)

    document.refresh_from_db()
    image_file = document.image or document.thumbnail
    if not image_file:
        return {"summary": "", "items": []}
    image_b64 = base64.b64encode(image_file.file.read()).decode()

    resp = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            },
        ],
    )
    try:
        content = resp.output_text
        return json.loads(content)
    except Exception:
        return {"summary": content if "content" in locals() else "", "items": []}


INTERACTIONS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "additionalProperties": False,
                "required": ["type", "label", "bbox"],
            },
        },
    },
    "additionalProperties": False,
    "required": ["interactions"],
}


def get_document_interactions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        print("No OpenAI API key found")
        return []
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a study assistant. You are given the textual description of a whiteboard, "
        "which includes the items drawn on it and their bounding boxes. Your task is to evaluate the "
        "contents and identify potential interactions to show the user. Interactions can be of types: "
        + "\n".join(
            [
                "- 'draw_graph': suggest drawing a graph based on data present or near a function definition",
                "- 'calculate': suggest performing a calculation based on numbers or formulas present",
                "- 'define': suggest defining a term or concept mentioned",
                "- 'summarize': suggest summarizing a section of text or a concept explained",
                "- 'translate': suggest translating text if multiple languages are detected",
                "- 'hint': suggest providing a hint for a problem or question posed",
                "- 'feedback': suggest giving feedback on some content, for example an equation step that needs correction",
            ]
        )
        + "\n\n"
        "For each interaction, provide the type, a brief label, and the bounding box [x1,y1,x2,y2]. You should use the bounding "
        "boxes provided in the analysis to anchor your interactions. "
        "Return a JSON with an array 'interactions'"
    )

    resp = client.responses.create(
        model="gpt-5",
        instructions=prompt,
        input=[
            {
                "role": "user",
                "content": json.dumps(analysis)
                + "\n\n"
                + "Your answer must be in json.",
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "schema": INTERACTIONS_JSON_SCHEMA,
                "name": "interactions_format",
            }
        },
    )
    try:
        content = resp.output_text
        return json.loads(content)["interactions"]
    except Exception:
        return []


def compute_detected_boxes_for_document(document) -> List[Tuple[int, int, int, int]]:
    """Best-effort detection of boxes using the image/thumbnail saved on the document."""
    try:
        if document.image:
            with document.image.open("rb") as f:
                image = Image.open(f).convert("RGB")
                return preprocess_thumbnail_for_boxes(image)
        if document.thumbnail:
            with document.thumbnail.open("rb") as f:
                image = Image.open(f).convert("RGB")
                return preprocess_thumbnail_for_boxes(image)
    except Exception as e:
        print("Error in box detection:", e)
    return []


def compute_analysis_for_document(document) -> Dict[str, Any]:
    """Compute and persist analysis for a document and return it."""
    detected_boxes = compute_detected_boxes_for_document(document)
    prompt = build_prompt(detected_boxes, document.data or {})
    analysis = get_document_analysis(prompt, document)
    document.analysis = analysis
    document.save(update_fields=["analysis", "updated_at"])
    return analysis


def compute_interactions_for_document(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    return get_document_interactions(analysis)


def run_analysis_pipeline(document) -> None:
    """Backward-compatible wrapper. Computes analysis then interactions. Left in place for REST path."""
    analysis = compute_analysis_for_document(document)
    interactions = get_document_interactions(analysis)
    # For now we don't persist interactions on the model; they're ephemeral
    print("Interactions:", interactions)
