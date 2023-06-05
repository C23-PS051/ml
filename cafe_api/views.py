from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from .models import Cafe
from .serializers import CafeSerializer
from firebase_admin import firestore


class CafeListAPIView(APIView):
    # add permission to check if user is authenticated
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        cafes = Cafe.objects.filter(user = request.user.id)
        serializer = CafeSerializer(cafes, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    # 2. Create
    def post(self, request, *args, **kwargs):
        data = {
            'name': request.data.get('name'),
            'user': request.user.id
        }
        serializer = CafeSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

