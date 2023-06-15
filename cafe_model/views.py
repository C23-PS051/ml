from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings
from .utils import predict

import numpy as np
import json


class GenerateCafeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        body_unicode = request.body.decode('utf-8')
        json_body = json.loads(body_unicode)

        new_user_id = 123 # placeholder
        new_is_male = json_body['new_is_male']
        new_age_group = json_body['new_age_group']
        new_price_category = json_body['new_price_category']
        new_24hrs = json_body['new_24hrs']
        new_outdoor = json_body['new_outdoor']
        new_smoking_area = json_body['new_smoking_area']
        new_parking_area = json_body['new_parking_area']
        new_pet_friendly = json_body['new_pet_friendly']
        new_wifi = json_body['new_wifi']
        new_indoor = json_body['new_indoor']
        new_live_music = json_body['new_live_music']
        new_takeaway = json_body['new_takeaway']
        new_kid_friendly = json_body['new_kid_friendly']
        new_alcohol = json_body['new_alcohol']
        new_in_mall = json_body['new_in_mall']
        new_toilets = json_body['new_toilets']
        new_reservation = json_body['new_reservation']
        new_vip_room = json_body['new_vip_room']

        user_vec = np.array([[new_is_male, new_age_group, new_price_category, new_24hrs, new_outdoor, new_smoking_area, new_parking_area, new_pet_friendly,
                            new_wifi, new_indoor, new_live_music, new_takeaway, new_kid_friendly, new_alcohol, new_in_mall, 
                            new_toilets, new_reservation, new_vip_room]])

        result = predict(settings.ML_VAR["model"], user_vec, settings.ML_VAR["cafe_trial"], settings.ML_VAR["scalerUser"], settings.ML_VAR["scalerItem"], settings.ML_VAR["scalerTarget"], settings.ML_VAR["cafe_data"], settings.ML_VAR["u_s"], settings.ML_VAR["c_s"])


        result_json = json.loads(result.to_json(orient="index"))

        return Response({"result": result_json}, status=status.HTTP_200_OK)

