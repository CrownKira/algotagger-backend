from django.db import models


class Prediction(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    model_used = models.CharField(max_length=100)
    prediction_result = models.JSONField()
