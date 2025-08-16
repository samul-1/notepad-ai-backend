from django.urls import path
from . import views


urlpatterns = [
    path(
        "documents/",
        views.DocumentListCreateView.as_view(),
        name="document-list-create",
    ),
    path(
        "documents/<int:pk>/",
        views.DocumentRetrieveUpdateView.as_view(),
        name="document-detail",
    ),
    path(
        "documents/<int:pk>/thumbnail/",
        views.DocumentThumbnailUploadView.as_view(),
        name="document-thumbnail",
    ),
]
