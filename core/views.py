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
                # "array": 0.1,
                # "string": 0.05,
                # "linked_list": 0.02,
                # "tree": 0.5,
                # "graph": 0.03,
                # "hash_table": 0.15,
                # "heap": 0.02,
                # "stack": 0.08,
                # "queue": 0.03,
                # "trie": 0.02,
                "array": 0.1,
                "string": 0.05,
                "dynamic_programming": 0.02,
                "math": 0.5,
                "hash_table": 0.03,
                "greedy": 0.15,
                "sorting": 0.02,
                "depth_first_search": 0.08,
                "breadth_first_search": 0.03,
                "binary_search": 0.02,
            }

            # Create the Prediction object and save it
            prediction = Prediction(
                title=serializer.validated_data["title"],
                description=serializer.validated_data["description"],
                model_used=serializer.validated_data["model_used"],
                prediction_result=dummy_prediction,
            )
            prediction.save()

            # Serialize the saved Prediction object and return the response
            response_serializer = PredictionSerializer(prediction)
            return Response(
                response_serializer.data, status=status.HTTP_201_CREATED
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
