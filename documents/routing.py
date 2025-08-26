from django.urls import path
from .consumers import DocumentConsumer

websocket_urlpatterns = [
    path("ws/documents/<int:doc_id>/", DocumentConsumer.as_asgi()),
]
