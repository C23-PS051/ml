from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CafeResult
from .serializers import CafeResultSerializer
# from firebase_admin import firestore

from numpy import loadtxt
from tensorflow.keras.models import load_model


class GenerateCafeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        # load model
        model = load_model("./models/model.h5")
        # summarize model.
        model.summary()
        # load dataset
        dataset = loadtxt("./models/pima-indians-diabetes.csv", delimiter=",")
        # split into input (X) and output (Y) variables
        X = dataset[:,0:8]
        Y = dataset[:,8]
        # evaluate the model
        score = model.evaluate(X, Y, verbose=0)

        result = "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)

        data = {
            'result': result,
        }
        serializer = CafeResultSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(None, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

