from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()