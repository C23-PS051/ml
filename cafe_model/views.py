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
        new_price_category = json_body.get("new_price_category", "$$")
        new_24hrs = json_body.get("new_24hrs", 0)
        new_outdoor = json_body.get("new_outdoor", 1)
        new_smoking_area = json_body.get("new_smoking_area", 0)
        new_parking_area = json_body.get("new_parking_area", 0)
        new_pet_friendly = json_body.get("new_pet_friendly", 0)
        new_wifi = json_body.get("new_wifi", 0)
        new_indoor = json_body.get("new_indoor", 1)
        new_live_music = json_body.get("new_live_music", 0)
        new_takeaway = json_body.get("new_takeaway", 0)
        new_kid_friendly = json_body.get("new_kid_friendly", 1)
        new_alcohol = json_body.get("new_alcohol", 0)
        new_in_mall = json_body.get("new_in_mall", 0)
        new_toilets = json_body.get("new_toilets", 0)
        new_reservation = json_body.get("new_reservation", 0)
        new_vip_room = json_body.get("new_vip_room", 0)

        user_vec = np.array([[new_is_male, new_age_group, new_price_category, new_24hrs, new_outdoor, new_smoking_area, new_parking_area, new_pet_friendly,
                            new_wifi, new_indoor, new_live_music, new_takeaway, new_kid_friendly, new_alcohol, new_in_mall, 
                            new_toilets, new_reservation, new_vip_room]])

        result = predict(settings.ML_VAR["model"], user_vec, settings.ML_VAR["cafe_trial"], settings.ML_VAR["scalerUser"], settings.ML_VAR["scalerItem"], settings.ML_VAR["scalerTarget"], settings.ML_VAR["cafe_data"], settings.ML_VAR["u_s"], settings.ML_VAR["c_s"])


        result_json = json.loads(result.to_json(orient="index"))

        return Response({"result": result_json}, status=status.HTTP_200_OK)

