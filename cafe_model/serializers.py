from rest_framework import serializers
from .models import CafeResult

class CafeResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = CafeResult
        fields = ["result", "timestamp"]