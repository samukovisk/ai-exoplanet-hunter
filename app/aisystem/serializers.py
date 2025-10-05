from rest_framework import serializers

class ExoplanetFileUploadSerializer(serializers.Serializer):
    file = serializers.FileField()