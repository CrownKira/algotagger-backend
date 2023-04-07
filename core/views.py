from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer


@api_view(["POST"])
def predict_data_structure(request):
    if request.method == "POST":
        serializer = PredictionSerializer(data=request.data)

        if serializer.is_valid():
            # Perform your actual classification here
            dummy_prediction = {
                "array": 0.1,
                "string": 0.05,
                "linked_list": 0.02,
                "tree": 0.5,
                "graph": 0.03,
                "hash_table": 0.15,
                "heap": 0.02,
                "stack": 0.08,
                "queue": 0.03,
                "trie": 0.02,
            }

            serializer.save(prediction_result=dummy_prediction)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
