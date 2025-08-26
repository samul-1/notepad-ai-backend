import sys
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .models import Document
from .serializers import DocumentSerializer
from .workers import run_analysis_pipeline


class DocumentListCreateView(generics.ListCreateAPIView):
    queryset = Document.objects.all().order_by("-updated_at")
    serializer_class = DocumentSerializer
    parser_classes = [JSONParser]


class DocumentRetrieveUpdateView(generics.RetrieveUpdateAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = [JSONParser]


class DocumentThumbnailUploadView(generics.UpdateAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser]

    def patch(self, request, *args, **kwargs):
        document = get_object_or_404(Document, pk=kwargs["pk"])
        file_obj = request.FILES.get("thumbnail")
        if not file_obj:
            return Response(
                {"detail": "No thumbnail provided"}, status=status.HTTP_400_BAD_REQUEST
            )
        # Save to both image and thumbnail for backward compatibility
        document.image.save(file_obj.name, file_obj, save=False)
        document.thumbnail.save(file_obj.name, file_obj, save=True)
        try:
            run_analysis_pipeline(document)
        except Exception:
            pass
        return Response(DocumentSerializer(document).data)


# Create your views here.
