import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn

def encode_categorical_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = OneHotEncoder()
    encoded_categorical_df = one_hot_encoder.fit_transform(dataframe[categorical_columns]).toarray()
    df = pd.DataFrame(encoded_categorical_df)
    df.index += 1
    return df

def get_one_encoder(dataframe: pd.DataFrame) -> sklearn.preprocessing._encoders.OneHotEncoder:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(dataframe[categorical_columns])
    return one_hot_encoder

def feature_engineer(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoded_categorical_df = encode_categorical_data(dataframe)
    numerical_columns = list(dataframe.select_dtypes(include='number').columns)
    continous_df = dataframe[numerical_columns]
    return pd.concat([continous_df, pd.DataFrame(encoded_categorical_df)], axis=1)