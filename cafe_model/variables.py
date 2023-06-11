from .utils import load_ml_model, load_cafe_data, load_cafe_trial, load_df, prepare_modelling, preprocess

df_cafe, df_user, y_label = load_df()
df_cafe, df_user = preprocess(df_cafe, df_user)
cafe_train, cafe_test, user_train, user_test, y_train, y_test, num_user_features, num_cafe_features, u_s, c_s, scalerItem, scalerUser, scalerTarget = prepare_modelling(df_cafe, df_user, y_label)
model = load_ml_model(cafe_train, cafe_test, user_train, user_test, y_train, y_test, num_user_features, num_cafe_features, u_s, c_s)
cafe_trial = load_cafe_trial()
cafe_data = load_cafe_data()

ML_VAR = {
    "model": model,
    "cafe_trial": cafe_trial, 
    "scalerUser": scalerUser, 
    "scalerItem": scalerItem, 
    "scalerTarget": scalerTarget, 
    "cafe_data": cafe_data, 
    "u_s": u_s, 
    "c_s": c_s,
}