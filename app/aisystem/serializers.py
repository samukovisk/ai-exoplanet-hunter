from rest_framework import serializers

class ExoplanetDataSerializer(serializers.Serializer):
    radius = serializers.FloatField()
    mass = serializers.FloatField()
    orbital_period = serializers.FloatField()
    temperature = serializers.FloatField()