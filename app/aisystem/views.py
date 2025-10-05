import csv
import io
import pandas as pd
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import ExoplanetFileUploadSerializer

def classify_exoplanet(data):
    radius = float(data.get('radius', 0))
    mass = float(data.get('mass', 0))
    orbital_period = float(data.get('orbital_period', 0))
    temperature = float(data.get('temperature', 0))

    if radius > 1.5 and mass > 1.0:
        return "Real Exoplanet"
    elif radius < 0.5:
        return "False Positive"
    else:
        return "Candidate"

@api_view(['POST'])
@parser_classes([MultiPartParser])
def classify_view(request):
    serializer = ExoplanetFileUploadSerializer(data=request.data)
    if serializer.is_valid():
        uploaded_file = serializer.validated_data['file']
        filename = uploaded_file.name.lower()

        results = []

        try:
            if filename.endswith('.csv'):
                decoded_file = uploaded_file.read().decode('utf-8')
                io_string = io.StringIO(decoded_file)
                reader = csv.DictReader(io_string)
                for row in reader:
                    classification = classify_exoplanet(row)
                    results.append({
                        "input": row,
                        "classification": classification
                    })

            elif filename.endswith('.xls') or filename.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                for _, row in df.iterrows():
                    data = row.to_dict()
                    classification = classify_exoplanet(data)
                    results.append({
                        "input": data,
                        "classification": classification
                    })
            else:
                return Response({"error": "Unsupported file format. Please upload a .csv or .xls/.xlsx file."},
                                status=status.HTTP_400_BAD_REQUEST)

            return Response({"results": results}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
