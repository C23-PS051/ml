from django.urls import include, re_path
from django.urls import path, include
from .views import (
    GenerateCafeAPIView
)

urlpatterns = [
    path('', GenerateCafeAPIView.as_view()),
]