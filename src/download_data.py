#!/usr/bin/env python
# Created by "Thieu" at 15:46, 01/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from sklearn.datasets import fetch_openml
from mafese import get_dataset
from sklearn.preprocessing import LabelEncoder


def get_phoneme_binary(verbose=False):
    data = fetch_openml(name='phoneme', version=1, as_frame=True, parser="auto")
    X, y = data.data, data.target.astype(int)

    # encoder
    label_encoder = LabelEncoder()
    # Fit and transform y
    y_encoded = label_encoder.fit_transform(y)

    df = pd.concat([X, pd.Series(y_encoded)], axis=1)
    df.to_csv("../data/clean/phoneme.csv", index=False)

    if verbose:
        print(f"\nData: Phoneme Binary Classification")
        print(f"X shape: {X.shape}, y shape: {y.shape}")


def get_aggregation(verbose=False):
    data = get_dataset("aggregation")
    # Split the dataset into training and testing sets

    df = pd.concat([pd.DataFrame(data.X), pd.Series(data.y)], axis=1)
    df.to_csv("../data/clean/aggregation.csv", index=False)

    if verbose:
        print(f"\nData: Phoneme Binary Classification")
        print(f"X shape: {data.X.shape}, y shape: {data.y.shape}")


def get_wine_multiclass(verbose=False):
    data = fetch_openml(name='wine', version=1, as_frame=True, parser="auto")
    X, y = data.data, data.target.astype(int)

    # encoder
    label_encoder = LabelEncoder()
    # Fit and transform y
    y_encoded = label_encoder.fit_transform(y)

    df = pd.concat([X, pd.Series(y_encoded)], axis=1)
    df.to_csv("../data/clean/wine.csv", index=False)

    if verbose:
        print(f"\nData: Wine Multiclass Classification")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

