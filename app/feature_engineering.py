import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn

def encode_categorical_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = OneHotEncoder()
    encoded_categorical_df = one_hot_encoder.fit(dataframe[categorical_columns])
    return encoded_categorical_df.transform(dataframe[categorical_columns])

def get_one_encoder(dataframe: pd.DataFrame) -> sklearn.preprocessing._encoders.OneHotEncoder:
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(dataframe[categorical_columns])
    return one_hot_encoder