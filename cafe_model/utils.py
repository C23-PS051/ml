import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

def load_df():
    df_cafe = pd.read_csv('./models/cafe_sorted.csv')
    df_user = pd.read_csv('./models/user_sorted.csv')
    y_label = pd.read_csv('./models/y_label_new.csv')
    df_cafe['cafe_id'] = 'cafe' + df_cafe['cafe_id'].astype(str)
    df_user['user_id'] = 'user' + df_user['user_id'].astype(str)
    return df_cafe, df_user, y_label


def preprocess(df_cafe, df_user):
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

    return df_cafe, df_user

def prepare_modelling(df_cafe, df_user, y_label):
    # I make a copy of each dataframe to make it easy to keep track
    user_train = df_user.copy()
    cafe_train = df_cafe.copy()
    y_train = y_label.copy()

    #drop irrelevant columns
    user_train = user_train.drop(['user_id', 'age', 'name'], axis=1)
    user_train.head()

    #drop irrelevant columns
    cafe_train = cafe_train.drop(['cafe_id', 'cafe_name', 'alamat', 'review', 'fasilitas'], axis=1)
    cafe_train.head()

    # Reshape the DataFrame to have all elements in a single column
    y_train = np.array(y_label).reshape(-1)
    user_train = np.array(user_train)
    cafe_train = np.array(cafe_train)

    num_user_features = user_train.shape[1] - 3 #remove user_id,	is_male,	age_group during training
    num_cafe_features = cafe_train.shape[1] - 1 #remove cafe_id during training
    u_s = 3  # start of columns to use in training, user
    c_s = 1  # start of columns to use in training, items

    # scale training data
    # cafe_train_unscaled = cafe_train
    # user_train_unscaled = user_train
    # y_train_unscaled = y_train

    scalerItem = StandardScaler()
    scalerItem.fit(cafe_train)
    cafe_train = scalerItem.transform(cafe_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))
    y_train = scalerTarget.transform(y_train.reshape(-1, 1))
    #ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

    cafe_train, cafe_test = train_test_split(cafe_train, train_size=0.80, shuffle=True, random_state=1)
    user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
    y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)

    return cafe_train, cafe_test, user_train, user_test, y_train, y_test, num_user_features, num_cafe_features, u_s, c_s, scalerItem, scalerUser, scalerTarget

def load_ml_model(cafe_train, cafe_test, user_train, user_test, y_train, y_test, num_user_features, num_cafe_features, u_s, c_s):
    num_outputs = 32
    tf.random.set_seed(1)
    user_NN = tf.keras.models.Sequential([     
        tf.keras.layers.Dense(units=128, activation='relu', name='l1'),
        tf.keras.layers.Dense(units=64, activation='relu', name='l2'),
        tf.keras.layers.Dense(units=num_outputs, activation='linear', name='l3')   
    ])

    item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu', name='l1'),
        tf.keras.layers.Dense(units=64, activation='relu', name='l2'),
        tf.keras.layers.Dense(units=num_outputs, activation='linear', name='l3') 
    ])

    # create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features))
    vu = user_NN(input_user)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    # create the item input and point to the base network
    input_item = tf.keras.layers.Input(shape=(num_cafe_features))
    vm = item_NN(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    # compute the dot product of the two vectors vu and vm
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    # specify the inputs and output of the model
    model = tf.keras.Model([input_user, input_item], output)

    model.summary()

    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
                loss=cost_fn)

    tf.random.set_seed(1)
    history = model.fit([user_train[:, u_s:], cafe_train[:, c_s:]], y_train,  epochs=30)

    model.evaluate([user_test[:, u_s:], cafe_test[:, c_s:]], y_test)

    return model

def load_cafe_trial():
    cafe_trial = pd.read_csv('./models/cafe_one_hot.csv')

    #encode price range
    label_encoder = LabelEncoder()
    cafe_trial['region'] = label_encoder.fit_transform(cafe_trial['region'])
    dollar_list = sorted(cafe_trial['kategori_harga'].unique())

    ordinal_encoder = OrdinalEncoder(categories=[dollar_list])

    encoded_data = ordinal_encoder.fit_transform(cafe_trial[['kategori_harga']])

    # dollar signs
    cafe_trial['kategori_harga'] = encoded_data

    #drop strings column
    cafe_trial = cafe_trial.drop(['cafe_id,', 'nama', 'alamat', 'review', 'fasilitas'], axis=1)
    
    return cafe_trial

def load_cafe_data():
    cafe_data = pd.read_csv('./models/cafe_one_hot.csv')
    return cafe_data


def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

def predict(model, user_vec, cafe_trial, scalerUser, scalerItem, scalerTarget, cafe_data, u_s, c_s):
    # generate and replicate the user vector to match the number movies in the data set.
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

    return sorted_items[:10]
    #print_pred_cafes(sorted_ypu, sorted_items, cafe_dict, maxcount = 10)