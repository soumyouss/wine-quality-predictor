import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


def compute_model_metrics(model: Pipeline, x: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
    predictions = model.predict(X=x)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return dict(rmse=rmse, mae=mae, r2=r2)
