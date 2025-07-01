#!/usr/bin/env python
# Created by "Thieu" at 15:26, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config


def to_numpy_array(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Input must be a DataFrame, Series, or ndarray. Got {type(x)} instead.")


def prepare_data(X, y, data_name=None, verbose=False):
    """
    Prepares the dataset by standardizing features and encoding the target variable if necessary.

    Parameters:
    - X: Features (numpy array or pandas DataFrame)
    - y: Target variable (numpy array or pandas Series)
    - data_name: Name of the dataset for logging purposes
    - encode: Whether to encode categorical target variables
    - verbose: Whether to print detailed information about the dataset

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets
    """
    # Convert X to a DataFrame if it's a numpy array
    X = to_numpy_array(X)
    y = to_numpy_array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=Config.TEST_SIZE, random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: {data_name}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_phoneme(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['target']), df['target'], data_name="Phoneme Binary Classification", verbose=verbose)


def get_rice(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['Class']), df['Class'], data_name="Rice Binary Classification", verbose=verbose)


def get_aggregation(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['target']), df['target'], data_name="Aggregation Multi-Class Classification", verbose=verbose)


def get_wine(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['target']), df['target'], data_name="Wine Multi-Class Classification", verbose=verbose)


def get_concrete(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['concrete_compressive_strength']), df['concrete_compressive_strength'], data_name="Concrete Regression", verbose=verbose)


def get_bike(path, verbose=False):
    df = pd.read_csv(path)
    return prepare_data(df.drop(columns=['count']), df['count'], data_name="Bike Sharing Demand Regression", verbose=verbose)



#
#
# # Helper function to standardize datasets and encode categorical target if needed
# def preprocess_data(X, y, data_name=None, encode=False, verbose=False):
#     # Convert X to a DataFrame if it's a numpy array
#     if isinstance(X, np.ndarray):
#         X = pd.DataFrame(X)
#
#     # Ensure y is a Series (in case it's passed as numpy array)
#     y = pd.Series(y)
#
#     # Combine X and y into a single DataFrame to drop NaN rows in both
#     data = X.copy()
#     data['target'] = y
#
#     # Remove rows with NaN values
#     data = data.dropna()
#
#     # Separate X and y after dropping NaNs
#     X = data.drop(columns=['target'])
#     y = data['target']
#
#     # Identify categorical and numeric columns
#     categorical_cols = X.select_dtypes(include=["object", "category"]).columns
#     numeric_cols = X.select_dtypes(include=['number']).columns
#
#     # Define transformers for numeric and categorical features
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#         ]
#     )
#
#     # Apply transformations
#     X_processed = preprocessor.fit_transform(X)
#
#     # Check if target is categorical and encode it if necessary
#     y = y.values
#     # if y.dtype == 'object' or y.dtype.name == 'category':  # Object type, usually indicating non-numeric labels
#     if encode:
#         le = LabelEncoder()
#         y = le.fit_transform(y)
#
#     X_train, X_test, y_train, y_test = train_test_split(X_processed, y,
#                                                         test_size=Config.TEST_SIZE, random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: {data_name}")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_phoneme_binary(verbose=False):
#     data = fetch_openml(name='phoneme', version=1, as_frame=True, parser="auto")
#     X, y = data.data, data.target.astype(int)
#     df = pd.concat([X, y], axis=1)
#     df.to_csv("data/phoneme_binary.csv", index=False)
#
#
# def get_magic(verbose=False):
#     # Magic Gamma Telescope dataset (Classification)
#     df = fetch_openml(name='MagicTelescope', version=2, as_frame=True, parser="auto")
#     return preprocess_data(df.data, df.target, data_name="Magic", encode=True, verbose=verbose)
#
# def get_california_housing(verbose=False):
#     df = fetch_california_housing()
#     return preprocess_data(df.data, df.target, data_name="California", encode=False, verbose=verbose)
#
#
# def get_bike_sharing_demand(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bike-sharing-demand-eda-regression-90-acc
#     # https://www.kaggle.com/code/behradkarimi/check-some-regression-model-and-find-the-best-one
#     # bike_sharing = fetch_ucirepo(id=275)
#
#     df = pd.read_csv(path)
#     X = df.drop(['count', 'year_2012', 'season_spring', 'season_summer', 'season_winter', 'hour_en'], axis=1).values
#     y = df['count'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Bike Sharing Demand")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         # print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_bank_marketing(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bank-marketing-eda-and-classification
#     # adult = fetch_ucirepo(id=222)
#
#     df = pd.read_csv(path)
#     X = df.drop('deposit', axis=1).values
#     y = df['deposit'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Bank Marketing")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_bankruptcy(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bankruptcy-classification-ml-95-acc
#     # taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)
#
#     df = pd.read_csv(path)
#     X = df.drop('Bankrupt?', axis=1).values
#     y = df['Bankrupt?'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Bankruptcy")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_car_evaluate(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/decision-tree-classification-car-evaluate
#     # car_evaluation = fetch_ucirepo(id=19)
#     # encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
#
#     df = pd.read_csv(path)
#     X = df.drop('class', axis=1).values
#     y = df['class'].values
#
#     y = LabelEncoder().fit_transform(y)
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Car Evaluation")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_letter_recognition(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/adityaw2604/support-vector-machine-92/input
#     # letter_recognition = fetch_ucirepo(id=59)
#
#     df = pd.read_csv(path)
#     X = df.drop('letter', axis=1).values
#     y = df['letter'].values
#     y = LabelEncoder().fit_transform(y)
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Letter Recognition")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_mushroom(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/mushroom-analysis-and-classification-accu-100
#     # mushroom = fetch_ucirepo(id=73)
#
#     df = pd.read_csv(path)
#     X = df.drop('edibility', axis=1).values
#     y = df['edibility'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Mushroom")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# def get_rice(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/rice-classification-ml-90-accuracy
#     # rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
#     ## Rice (Cammeo and Osmancik)
#
#     df = pd.read_csv(path)
#     X = df.drop('Class', axis=1).values
#     y = df['Class'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Rice")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test

#
# def get_ccpp(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/combined-cycle-power-plant-forecast-ml-93-acc
#     # combined_cycle_power_plant = fetch_ucirepo(id=294)
#
#     df = pd.read_csv(path)
#     X = df.drop('AT', axis=1).values
#     y = df['AT'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Combined Cycle Power Plant")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_concrete(path, verbose=False):
#     # (1030, 9)
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/concrete-compression-forecasting-ml-95-acc
#
#     df = pd.read_csv(path)
#     X = df.drop('concrete_compressive_strength', axis=1).values
#     y = df['concrete_compressive_strength'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Concrete Compression Strength")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_air_quality(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/air-quality-with-ml-85-acc-dl-98-acc
#     # air_quality = fetch_ucirepo(id=360)
#
#     df = pd.read_csv(path)
#     X = df.drop('AH', axis=1).values
#     y = df['AH'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Air Quality")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_superconductivty(path, verbose=False):
#     # (21263, 81)
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/superconductivty-forecasting-ml-88-acc
#     # superconductivty_data = fetch_ucirepo(id=464)
#
#     df = pd.read_csv(path)
#     X = df.drop('critical_temp', axis=1).values
#     y = df['critical_temp'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Superconductivty")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_spambase(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/email-spam-classification-ml-model-97-accuracy
#     # spambase = fetch_ucirepo(id=94)
#
#     df = pd.read_csv(path)
#     X = df.drop('spam', axis=1).values
#     y = df['spam'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Spambase")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_titanic(path, verbose=False):
#     ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/titanic-classification-ml-90-accuracy
#
#     df = pd.read_csv(path)
#     X = df.drop('Survived', axis=1).values
#     y = df['Survived'].values
#
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
#                                                         random_state=Config.SEED_SPLIT_DATA)
#     if verbose:
#         print(f"\nData: Titanic")
#         print(f"X shape: {X.shape}, y shape: {y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test
#
#
# def get_blobs(verbose=False):
#     data = get_dataset("blobs")
#     # Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = (
#         data.split_train_test(test_size=Config.TEST_SIZE, random_state=Config.SEED_SPLIT_DATA, inplace = False))
#
#     if verbose:
#         print(f"\nData: Blobs")
#         print(f"X shape: {data.X.shape}, y shape: {data.y.shape}")
#         print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
#         print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# get_blobs(True)
# get_aggregation(verbose=True)
