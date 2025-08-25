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

    # def perform_update(self, serializer):
    #     document = serializer.save()
    #     try:
    #         run_analysis_pipeline(document)
    #     except Exception:
    #         # best-effort; don't fail the save
    #         print("Error running analysis pipeline:", sys.exc_info())
    #         pass


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
        document.thumbnail.save(file_obj.name, file_obj, save=True)
        try:
            print("Running analysis pipeline...")
            run_analysis_pipeline(document)
        except Exception:
            pass
        return Response(DocumentSerializer(document).data)


# Create your views here.
