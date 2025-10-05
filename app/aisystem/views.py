from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import ExoplanetFileUploadSerializer
from .classifier.predictor import ExoplanetPredictor  # Import the predictor

import csv
import io
import pandas as pd
import os
import tempfile


# Initialize the predictor (paths adjusted to Docker)
predictor = ExoplanetPredictor(
    model_path='/app/aisystem/classifier/xgboost_grid_best_model1.joblib',
    training_data_path='/app/aisystem/classifier/datasets/selected_features_exoplanets.csv'

)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def classify_view(request):
    serializer = ExoplanetFileUploadSerializer(data=request.data)
    if serializer.is_valid():
        uploaded_file = serializer.validated_data['file']
        filename = uploaded_file.name.lower()

        try:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Run batch prediction
            results_df = predictor.predict_batch(input_file=temp_file_path)

            # Convert results to JSON
            results_json = results_df.to_dict(orient='records')

            # Clean up temporary file
            os.remove(temp_file_path)

            return Response({"results": results_json}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
