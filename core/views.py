import random

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import PredictionSerializer
from .predictors.xgb.predictor import predict_xgb
from .predictors.distilbert.predictor import (
    predict_distilbert,
)
from .predictors.svm.predictor import (
    predict_svm,
)


@api_view(["POST"])
def predict_data_structure(request):
    if request.method == "POST":
        serializer = PredictionSerializer(data=request.data)

        if serializer.is_valid():
            title = serializer.validated_data["title"]
            description = serializer.validated_data["description"]
            model_used = serializer.validated_data["model_used"]

            if model_used.lower() == "xgboost":
                dummy_prediction = predict_xgb(title, description)
            elif model_used.lower() == "distilbert":
                dummy_prediction = predict_distilbert(title, description)
            elif model_used.lower() == "svm":
                dummy_prediction = predict_svm(title, description)
            else:
                dummy_prediction = {
                    "array": random.random(),
                    "string": random.random(),
                    "dynamic_programming": random.random(),
                    "math": random.random(),
                    "hash_table": random.random(),
                    "greedy": random.random(),
                    "sorting": random.random(),
                    "depth_first_search": random.random(),
                    "breadth_first_search": random.random(),
                    "binary_search": random.random(),
                }

            response_data = {
                "title": title,
                "description": description,
                "model_used": model_used,
                "prediction_result": dummy_prediction,
            }

            return Response(response_data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
