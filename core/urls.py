from django.urls import path
from . import views

urlpatterns = [
    path(
        "predict/", views.predict_data_structure, name="predict_data_structure"
    ),
]
