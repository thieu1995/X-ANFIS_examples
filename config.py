#!/usr/bin/env python
# Created by "Thieu" at 19:56, 13/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
# BASE_PATH = Path.cwd().parent
BASE_PATH = Path.cwd()

class Config:

    PATH_READ = f"{BASE_PATH}/data/clean"
    PATH_SAVE = f"{BASE_PATH}/data/history"
    STATISTIC_FILE_NAME = "statistic-results.csv"
    FILE_LOSS = "df_loss.csv"
    FILE_RESULT = "df_result.csv"
    FILE_RESULT_STATS = "df_result_statistics.csv"
    FIGURE_SIZE = (10, 4.8)

    VERBOSE = True
    SEED_SPLIT_DATA = 42
    TEST_SIZE = 0.2
    OBJ_CLS = "F1S"
    OBJ_REG = "MSE"

    # 5404 samples, 5 features, 2 classes  GBell => Trapezoidal, Gaussian  => SShaped
    DATA01 = {
        "name": "phoneme",
        "n_rules": 12,
        "mf_class": "Trapezoidal", # Gaussian, Sigmoid, Trapezoidal, Triangular, Sigmoid, Bell, PiShaped, SShaped, GBell, ZShaped, Linear
        "vanishing_strategy": "prod"
    }

    # 13376 samples, 10 features, 2 classes
    DATA02 = {
        "name": "magic",
        "n_rules": 12,
        "mf_class": "Triangular", # Gaussian, Sigmoid, Trapezoidal, Triangular, Sigmoid, Bell, PiShaped, SShaped, GBell, ZShaped, Linear
        "vanishing_strategy": "prod"
    }

    # 788 samples, 3 features, 7 classes
    DATA03 = {
        "name": "aggregation",
        "n_rules": 8,
        "mf_class": "Gaussian",
        "vanishing_strategy": "prod"
    }

    # 1500 samples, 2 features, 3 classes - blobs
    # 178 samples, 13 features, 3 classes - wine  Triangular: 0.876
    DATA04 = {
        "name": "wine",
        "n_rules": 10,
        "mf_class": "Triangular", # Gaussian, Sigmoid, Trapezoidal, Triangular, Sigmoid, Bell, PiShaped, SShaped, GBell, ZShaped, Linear
        "vanishing_strategy": "prod"
    }

    # 17376 samples, 9 features,
    DATA05 = {
        "name": "bike",
        "n_rules": 15,
        "mf_class": "Triangular", # Gaussian, Sigmoid, Trapezoidal, Triangular, Sigmoid, Bell, PiShaped, SShaped, GBell, ZShaped, Linear
        "vanishing_strategy": "prod"
    }

    # 20K samples, 8 features
    DATA06 = {
        "name": "california",
        "n_rules": 15,
        "mf_class": "Triangular", # Gaussian, Sigmoid, Trapezoidal, Triangular, Sigmoid, Bell, PiShaped, SShaped, GBell, ZShaped, Linear
        "vanishing_strategy": "prod"
    }

    EPOCH = 500
    POP_SIZE = 20
    # LIST_SEEDS = [7, 8, 11, 15, 20, 21, 22, 23, 24, 27, 28, 30, 32, 35, 37, 39, 40, 41, 42, 45]
    # LIST_METRICS = ["PS", "RS", "NPV", "F1S", "F2S", "SS", "CKS", "GMS", "AUC", "LS", "AS"]
    LIST_SEEDS = [10, 15, 21, 24, 27, 29, 30, 35, 40, 42]
    # LIST_SEEDS = [10,]
    LIST_METRIC_CLS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
    LIST_METRIC_REG =  ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
    # LIST_MODEL_CC = ["SHADE-ANFIS", "L-SHADE-ANFIS", "IM-ARO-ANFIS", "QLE-SCA-ANFIS", "A-EO-ANFIS", "AIW-PSO-ANFIS"]
    LIST_MODEL_CC = ["GBO-ANFIS", "ARO-ANFIS", "RIME-ANFIS", "QLE-SCA-ANFIS", "AAEO-ANFIS", "WOA-ANFIS"]     # Models used to draw convergence chart
    # LIST_MODEL_CC = ["Adam-ANFIS", ]
    N_WORKERS = 10

    # EPOCH = 50
    # POP_SIZE = 20
    # LIST_SEEDS = [7]

    LIST_GD_MODELS = [
        {"name": "Adam-ANFIS", "class": "Adam", "paras": {}},  #
        {"name": "RMSprop-ANFIS", "class": "RMSprop", "paras": {}},  #
        {"name": "Adadelta-ANFIS", "class": "Adadelta", "paras": {}},       #
    ]

    LIST_BIO_MODELS = [
        {"name": "GBO-ANFIS", "class": "OriginalGBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "ARO-ANFIS", "class": "OriginalARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "RIME-ANFIS", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "QLE-SCA-ANFIS", "class": "QleSCA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "AAEO-ANFIS", "class": "AugmentedAEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SMA-ANFIS", "class": "DevSMA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"name": "AVOA-ANFIS", "class": "OriginalAVOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "AGTO-ANFIS", "class": "OriginalAGTO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "WOA-ANFIS", "class": "OriginalWOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},

        # {"name": "LDW-PSO-ANFIS", "class": "LDW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5
        # {"name": "CL-PSO-ANFIS", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 5.5 - 7.5
        # # {"name": "AGTO-ANFIS", "class": "OriginalAGTO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 13
        # # {"name": "AVOA-ANFIS", "class": "OriginalAVOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5    But getting warning divide by 0
        # {"name": "SMA-ANFIS", "class": "OriginalSMA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5
        # # {"name": "SOS-ANFIS", "class": "OriginalSOS", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 21 - 26
        # {"name": "GBO-ANFIS", "class": "OriginalGBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},          # 7
        # {"name": "PSS-ANFIS", "class": "OriginalPSS", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},           # 7
        # # {"name": "E-AEO-ANFIS", "class": "EnhancedAEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 14 - 16
        # # {"name": "AAEO-ANFIS", "class": "AugmentedAEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 14
        # {"name": "SADE-ANFIS", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},                 # 7
        # {"name": "CMA-ES-ANFIS", "class": "CMA_ES", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},             # 7
        # {"name": "SHADE-ANFIS", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 4.5
        # # {"name": "TLO-ANFIS", "class": "OriginalTLO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 15
        # # {"name": "QSA-ANFIS", "class": "OriginalQSA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 14
        # {"name": "EFO-ANFIS", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 0.33
        # # {"name": "M-EO-ANFIS", "class": "ModifiedEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 13
        # {"name": "RIME-ANFIS", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 8
        # # {"name": "MGTO-ANFIS", "class": "MGTO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 15
        # {"name": "HI-WOA-ANFIS", "class": "HI_WOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 6
        # # {"name": "SHO-ANFIS", "class": "OriginalSHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},   # 19 - 36
        # # {"name": "WMQI-MRFO-ANFIS", "class": "WMQIMRFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},    # Warning divide by 0,
        # {"name": "WOA-FOA-ANFIS", "class": "WhaleFOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},      # 6
    ]
