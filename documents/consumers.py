import json
import base64
import uuid
from typing import Any, Dict
from django.core.files.base import ContentFile
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import Document
from .workers import (
    compute_analysis_for_document,
    compute_interactions_for_document,
)


class DocumentConsumer(AsyncJsonWebsocketConsumer):
    group_name: str

    async def connect(self):
        self.doc_id = int(self.scope["url_route"]["kwargs"].get("doc_id"))
        self.group_name = f"document_{self.doc_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive_json(self, content: Dict[str, Any], **kwargs):
        event = content.get("event")
        if event == "document.update":
            await self.handle_document_update(content)

    @sync_to_async
    def _save_document_update(self, data: Dict[str, Any]):
        doc = Document.objects.get(pk=self.doc_id)
        # Save excalidraw data
        if "data" in data:
            doc.data = data["data"]
        # Save image/thumbnail if provided (base64 PNG)
        image_b64 = data.get("image_base64")
        updated_fields = ["data", "updated_at"]
        if image_b64:
            # image_b64 can be data URL; strip prefix if present
            prefix = "data:image/png;base64,"
            if image_b64.startswith(prefix):
                image_b64 = image_b64[len(prefix) :]
            raw_bytes = base64.b64decode(image_b64)
            uid = uuid.uuid4().hex[:8]
            doc.image.save(
                f"doc_{doc.id}_{uid}.png", ContentFile(raw_bytes), save=False
            )
            doc.thumbnail.save(
                f"thumb_{doc.id}_{uid}.png", ContentFile(raw_bytes), save=False
            )
            updated_fields.extend(["image", "thumbnail"])
        doc.save(update_fields=updated_fields)
        return doc

    async def handle_document_update(self, content: Dict[str, Any]):
        # 1) Save incoming changes
        doc = await self._save_document_update(content)
        # 2) Compute analysis and notify client when done
        analysis = await sync_to_async(compute_analysis_for_document)(doc)
        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": "document.analysis.done",
                "analysis": analysis,
            },
        )
        # 3) Compute interactions and notify
        interactions = await sync_to_async(compute_interactions_for_document)(analysis)
        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": "document.interactions",
                "interactions": interactions,
            },
        )

    async def document_analysis_done(self, event: Dict[str, Any]):
        await self.send_json(
            {"event": "document.analysis.done", "analysis": event["analysis"]}
        )

    async def document_interactions(self, event: Dict[str, Any]):
        await self.send_json(
            {"event": "document.interactions", "interactions": event["interactions"]}
        )
