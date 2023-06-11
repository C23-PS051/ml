from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CafeResult
from .serializers import CafeResultSerializer
# from firebase_admin import firestore

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
import numpy as np
from .utils import gen_user_vecs
import json


class GenerateCafeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        cafe_data = pd.read_csv('./models/cafe_data.csv')
        scalerUser = StandardScaler()
        scalerItem = StandardScaler()
        scalerTarget = MinMaxScaler()

        df_cafe = pd.read_csv('./models/cafe_sorted.csv')
        df_user = pd.read_csv('./models/user_sorted.csv')
        y_label = pd.read_csv('./models/y_label_new.csv')

        df_cafe.head()

        df_user.head()

        y_label.iloc[246:310]

        """#Data Preprocessing"""

        #encode region
        #0 - Central Jakarta
        #1 - East Jakarta
        #2 - North Jakarta
        #3 - South Jakarta
        #4 - West Jakarta
        label_encoder = LabelEncoder()
        df_cafe['region'] = label_encoder.fit_transform(df_cafe['region'])

        #one hot encoding facilities
        keywords = ['outdoor','smoking_area', 'parking_area', 'pet_friendly', 'wifi', 'indoor', 'live_music', 'takeaway',
                    'kid_friendly', 'alcohol', 'in_mall', 'toilets', 'reservation', 'vip_room']

        ttt = 'is_'
        for i in keywords:
            a = df_cafe['fasilitas'].str.find(i)
            df_cafe.loc[df_cafe[a >= 0].index, ttt+i] = 1
            
        df_cafe = df_cafe.fillna(0)

        #encode price range

        dollar_list = sorted(df_cafe['kategori_harga'].unique())

        ordinal_encoder = OrdinalEncoder(categories=[dollar_list])

        encoded_data = ordinal_encoder.fit_transform(df_cafe[['kategori_harga']])

        # dollar signs
        df_cafe['kategori_harga'] = encoded_data

        #encode price range

        dollar_user = sorted(df_user['price_category'].unique())

        ordinal_encoder = OrdinalEncoder(categories=[dollar_user])

        encoded_data = ordinal_encoder.fit_transform(df_user[['price_category']])

        # dollar signs
        df_user['price_category'] = encoded_data

        df_cafe['kategori_harga'] = df_cafe['kategori_harga'].astype(int)
        df_user['price_category'] = df_user['price_category'].astype(int)
        df_cafe.iloc[:, 8:] = df_cafe.iloc[:, 8:].astype(int)

        def age_classifier(x):
            if x <= 20:
                return 0 #15-20
            elif x <= 25 and x > 20:
                return 1 #21-25
            elif x <= 30 and x > 25:
                return 2 #26-30
            elif x <= 35 and x > 30:
                return 3 #31-35
            elif x <= 40 and x > 35:
                return 4 #36-40
            elif x <= 45 and x > 40:
                return 5 #41-45
            elif x <= 50 and x > 45:
                return 6 #46-50    
            else:
                return 7

        age_group = df_user['age'].apply(age_classifier)
        df_user.insert(4, 'age_group', age_group)

        df_cafe.head()

        df_user.head()

        """#CNN Modelling

        ##Preparing Data for Modelling
        """

        # I make a copy of each dataframe to make it easy to keep track
        user_train = df_user.copy()
        cafe_train = df_cafe.copy()
        y_train = y_label.copy()

        #drop irrelevant columns
        user_train = user_train.drop(['age', 'name', 'is_male', 'age_group'], axis=1)
        user_train.head()

        #drop irrelevant columns
        cafe_train = cafe_train.drop(['cafe_name', 'alamat', 'review', 'fasilitas'], axis=1)

        # Reshape the DataFrame to have all elements in a single column
        y_train = np.array(y_label).reshape(-1)
        user_train = np.array(user_train)
        cafe_train = np.array(cafe_train)

        num_user_features = user_train.shape[1] - 1 #remove user_id,	is_male,	age_group during training
        num_cafe_features = cafe_train.shape[1] - 1 #remove cafe_id during training
        uvs = 1  # user genre vector start
        cvs = 3  # item genre vector start
        u_s = 1  # start of columns to use in training, user
        c_s = 1  # start of columns to use in training, items

        # scale training data
        cafe_train_unscaled = cafe_train
        user_train_unscaled = user_train
        y_train_unscaled = y_train

        scalerItem.fit(cafe_train)
        cafe_train = scalerItem.transform(cafe_train)
        scalerUser.fit(user_train)
        user_train = scalerUser.transform(user_train)

        scalerTarget = MinMaxScaler((-1, 1))
        scalerTarget.fit(y_train.reshape(-1, 1))




        # load model
        model = load_model("./models/model.h5")
        # load dataset
        cafe_trial = pd.read_csv('./models/cafe_trial.csv')

        body_unicode = request.body.decode('utf-8')
        json_body = json.loads(body_unicode)

        new_user_id = json_body['new_user_id']
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

        user_vec = np.array([[new_user_id, new_price_category, new_24hrs, new_outdoor, new_smoking_area, new_parking_area, new_pet_friendly,
                            new_wifi, new_indoor, new_live_music, new_takeaway, new_kid_friendly, new_alcohol, new_in_mall, 
                            new_toilets, new_reservation, new_vip_room]])

        # generate and replicate the user vector to match the number cafe in the data set.
        user_vecs = gen_user_vecs(user_vec,len(cafe_trial))

        # scale our user and item vectors
        suser_vecs = scalerUser.transform(user_vecs)
        sitem_vecs = scalerItem.transform(cafe_trial)

        # make a prediction
        y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, c_s:]])

        # unscale y prediction 
        y_pu = scalerTarget.inverse_transform(y_p)

        # sort the results, highest prediction first
        sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_items = cafe_data.iloc[sorted_index]  #using unscaled vectors for display

        print(sorted_items)
        
        return Response(None, status=status.HTTP_200_OK)

