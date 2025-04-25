#!/usr/bin/env python
# Created by "Thieu" at 02:50, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from permetrics import ClassificationMetric, RegressionMetric
import pandas as pd
from pathlib import Path
import pickle


def get_metrics(problem="regression", y_true=None, y_pred=None, list_metrics=("RMSE", "MAE")):
    """
    Get evaluation metrics
    """
    if problem == "regression":
        cm = RegressionMetric(y_true, y_pred)
    else:
        cm = ClassificationMetric(y_true, y_pred)
    return cm.get_metrics_by_list_names(list_metrics)


def save_metrics(problem="regression", y_true=None, y_pred=None, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
    """
    Save evaluation metrics to csv file
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    results = get_metrics(problem, y_true, y_pred, list_metrics)
    df = pd.DataFrame.from_dict(results, orient='index').T
    df.to_csv(f"{save_path}/{filename}", index=False)


def save_model(model, save_path="history", filename="model.pkl"):
    """
    Save model to pickle file

    Parameters
    ----------
    save_path : saved path (relative path, consider from current executed script path)
    filename : name of the file, needs to have ".pkl" extension
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    if filename[-4:] != ".pkl":
        filename += ".pkl"
    pickle.dump(model, open(f"{save_path}/{filename}", 'wb'))
