import pandas as pd

def impute_categorical_missing_value(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_value = 'Missing Value'
    categorical_columns = list(dataframe.select_dtypes(include='object').columns)
    for column_name in categorical_columns:
        dataframe[column_name] = dataframe[column_name].fillna(new_value)
    return dataframe

def impute_continuous_missing_value(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_value = 0
    continous_columns = list(dataframe.select_dtypes(include='number').columns)
    for column_name in continous_columns:
        dataframe[column_name] = dataframe[column_name].fillna(new_value)
    return dataframe

def impute_missing_value(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoded_dataframe = impute_categorical_missing_value(dataframe)
    encoded_dataframe = impute_continuous_missing_value(encoded_dataframe)
    return encoded_dataframe

def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = dataframe.copy()
    return impute_missing_value(preprocessed_df)