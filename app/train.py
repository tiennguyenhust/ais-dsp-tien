from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from app.inference import inference
from app.preprocess import preprocess


def train(filepath: Path, models_dir: Path):
    X_train_processed, X_test, y_train, y_test = get_train_data(filepath, models_dir)
    model = get_fitted_model(X_train_processed, y_train, models_dir, 'RandomForestRegressor', RandomForestRegressor, **{'max_depth': 7, 'random_state': 0})
    y_pred = inference(models_dir, data=X_test)
    rmsle = compute_rmsle(y_test, y_pred)
    
    return model, rmsle 
    

def get_train_data(filepath: Path, models_dir: Path) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(filepath, index_col='Id')
    X, y = df.loc[:, df.columns != 'SalePrice'], df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_processed = preprocess(X_train, models_dir)
    
    return X_train_processed, X_test, y_train, y_test
    

def get_fitted_model(X_train: pd.DataFrame, y_train: pd.Series, models_dir: Path, model_name: str, model_class, **model_kwargs):
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    joblib.dump(model, models_dir / f'{model_name}.joblib')
    
    return model


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
    