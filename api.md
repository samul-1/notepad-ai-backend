# Backend API for Notepad AI Frontend

This document describes how the Excalidraw-based frontend can interact with the
Django backend to obtain real-time AI analysis and interaction suggestions.

## Endpoints

### `POST /documents/`
Create a new whiteboard document.

**Body**
```json
{
  "title": "Optional title",
  "data": {"elements": []} // raw Excalidraw JSON
}
```

### `GET /documents/`
List existing documents ordered by last update.

### `GET /documents/{id}/`
Retrieve a single document including analysis and suggested interactions.

### `PATCH /documents/{id}/`
Update the Excalidraw JSON for a document. Saving triggers the analysis
pipeline which calls the OpenAI Responses API and stores the result.

**Body**
```json
{
  "data": { ... Excalidraw JSON ... }
}
```

### `POST /documents/{id}/thumbnail/`
Upload a PNG thumbnail of the canvas. The image is used to compute bounding
boxes for the analysis.

**Form field**
- `thumbnail`: binary PNG blob

## Response payload
Each document serializes as:

```json
{
  "id": 1,
  "title": "Untitled",
  "data": { ... },
  "thumbnail": "/media/thumbnails/1.png",
  "analysis": {
    "summary": "High level description of the board",
    "items": [
      {"id": "e1", "type": "text", "text": "x+1=0", "bbox": [10,10,120,40]}
    ]
  },
  "interactions": [
    {
      "id": "hint1",
      "bbox": [10,10,120,40],
      "label": "Explain",
      "action": "explain_step"
    }
  ],
  "created_at": "...",
  "updated_at": "..."
}
```

### Interaction schema
Each element in `interactions` has:
- `id`: unique identifier for the interaction.
- `bbox`: `[x1, y1, x2, y2]` in canvas coordinates.
- `label`: short text to show in the UI.
- `action`: machine-readable string describing what to do when triggered.

Example actions:
- `explain_step` – provide a textual explanation for a step.
- `plot_function` – render the graph of a function.
- `fill_table` – fill in missing value in a truth table.
- `search_image` – fetch an image related to the nearby text.

## Frontend flow
1. Create or load a document via the endpoints above.
2. On each change, send the updated Excalidraw JSON with `PATCH /documents/{id}/`.
3. Periodically upload a thumbnail to `/documents/{id}/thumbnail/` so the backend
   can compute bounding boxes.
4. After each update, fetch the document (`GET /documents/{id}/`) and render the
   returned `interactions` as buttons or widgets at the specified `bbox`.
5. When the user activates an interaction, perform the action in the UI. Some
   actions may require additional calls to OpenAI (e.g. generating a hint or an
   image) which the frontend can handle separately.

## OpenAI Responses
The backend calls the OpenAI Responses API with structured output. The expected
schema is equivalent to:
```json
{
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "items": {"type": "array", "items": {"type": "object"}},
    "interactions": {"type": "array", "items": {"type": "object"}}
  },
  "required": ["summary", "items", "interactions"]
}
```
This is handled server-side; the frontend only consumes the result.

