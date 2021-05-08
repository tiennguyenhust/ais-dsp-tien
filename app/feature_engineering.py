import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn
import pickle

def save_one_encoder(dataframe: pd.DataFrame):
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    one_hot_encoder.fit(dataframe[categorical_columns])
    filename = '../data/one_encoder.sav'
    pickle.dump(one_hot_encoder, open(filename, 'wb'))

# def get_one_encoder(dataframe: pd.DataFrame) -> sklearn.preprocessing._encoders.OneHotEncoder:
#     categorical_columns = list(dataframe.select_dtypes(include='object').columns)
#     one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
#     one_hot_encoder.fit(dataframe[categorical_columns])
#     return one_hot_encoder

def get_one_encoder() -> sklearn.preprocessing._encoders.OneHotEncoder:
    filename = '../data/one_encoder.sav'
    return pickle.load(open(filename, 'rb'))

def encode_categorical_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = get_one_encoder()
    encoded_categorical_df = one_hot_encoder.transform(dataframe[categorical_columns]).toarray()
    df = pd.DataFrame(encoded_categorical_df)
    df.index += 1
    return df

def feature_engineer(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoded_categorical_df = encode_categorical_data(dataframe)
    numerical_columns = list(dataframe.select_dtypes(include='number').columns)
    continous_df = dataframe[numerical_columns]
    return pd.concat([continous_df, pd.DataFrame(encoded_categorical_df)], axis=1)