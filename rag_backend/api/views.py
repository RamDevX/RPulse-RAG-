from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuerySerializer
from .rag import rag_simple, rag_retriever, llm

class QueryAPIView(APIView):
    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']

            try:
                # Call your RAG pipeline
                answer = rag_simple(question, rag_retriever, llm)
                return Response({'answer': answer}, status=status.HTTP_200_OK)

            except Exception as e:
                return Response(
                    {"error": f"Error while generating answer: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

