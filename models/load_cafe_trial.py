import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

cafe_trial = pd.read_csv('./cafe_one_hot.csv')

#encode price range
label_encoder = LabelEncoder()
cafe_trial['region'] = label_encoder.fit_transform(cafe_trial['region'])
dollar_list = sorted(cafe_trial['kategori_harga'].unique())

ordinal_encoder = OrdinalEncoder(categories=[dollar_list])

encoded_data = ordinal_encoder.fit_transform(cafe_trial[['kategori_harga']])

# dollar signs
cafe_trial['kategori_harga'] = encoded_data

#drop strings column
cafe_trial = cafe_trial.drop(['nama', 'alamat', 'review', 'fasilitas'], axis=1)
cafe_trial.head()

cafe_trial.to_csv('cafe_trial.csv')