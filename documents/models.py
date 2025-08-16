from django.db import models
from django.utils import timezone


class Document(models.Model):
    title = models.CharField(max_length=255, default="Untitled")
    # Raw Excalidraw JSON data as text
    data = models.JSONField(default=dict, blank=True)
    # Generated thumbnail (PNG) of the whiteboard
    thumbnail = models.ImageField(upload_to="thumbnails/", null=True, blank=True)
    # Most recent AI analysis: description and element boxes
    analysis = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Document({self.id}) - {self.title}"


# Create your models here.
