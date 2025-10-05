from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import ExoplanetDataSerializer

# Dummy AI classifier function
def classify_exoplanet(data):
    # Replace this with your actual model logic
    radius = data['radius']
    mass = data['mass']
    orbital_period = data['orbital_period']
    temperature = data['temperature']

    # Simple mock logic
    if radius > 1.5 and mass > 1.0:
        return "Real Exoplanet"
    elif radius < 0.5:
        return "False Positive"
    else:
        return "Candidate"

@api_view(['POST'])
def classify_view(request):
    serializer = ExoplanetDataSerializer(data=request.data)
    if serializer.is_valid():
        classification = classify_exoplanet(serializer.validated_data)
        return Response({'classification': classification})
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
