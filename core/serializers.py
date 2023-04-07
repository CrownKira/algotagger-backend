from rest_framework import serializers
from .models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    prediction_result = serializers.JSONField(read_only=True)

    class Meta:
        model = Prediction
        fields = "__all__"
