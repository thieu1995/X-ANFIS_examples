#!/usr/bin/env python
# Created by "Thieu" at 06:30, 25/04/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed
from xanfis import DataTransformer, AnfisClassifier, GdAnfisClassifier, BioAnfisClassifier
from config import Config as cf
from src.data_utils import get_phoneme
from src.helper import get_metrics
from src.visualizer import draw_confusion_matrix, draw_boxplot, draw_convergence_chart


def run_trial(md_item, data, cf):
    # Function to train, test, and evaluate a model for a single seed
    X_train, X_test, y_train, y_test = data
    model, seed, model_name, has_loss = md_item

    # Train the model
    model.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    if has_loss == "no_loss":
        loss_train = [0] * cf.EPOCH
    else:
        loss_train = model.loss_train
    res_epoch_loss = [{"model_name": model_name, "seed": seed, "epoch": epoch + 1, "loss": loss} for epoch, loss in enumerate(loss_train)]

    # Predict and evaluate
    y_pred = model.predict(X_test)
    res = get_metrics(problem="classification", y_true=y_test, y_pred=y_pred, list_metrics=cf.LIST_METRIC_CLS)
    res_predict = {"model_name": model_name, "seed": seed, **res}

    dt_name = cf.DATA01['name'].capitalize()

    # Draw the figure
    draw_confusion_matrix(y_true=y_test, y_pred=y_pred,
                          figsize=(8, 6), title=f"Confusion matrix of {model_name} on {dt_name} dataset",
                          pathsave=f"{cf.PATH_SAVE}/{cf.DATA01['name']}/cm/cm_{model_name}_{seed}.png", verbose=False)
    return res_epoch_loss, res_predict


if __name__ == "__main__":
    ## Load data object
    # 788 samples, 2 features, 7 classes
    Path(f"{cf.PATH_SAVE}/{cf.DATA01['name']}").mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = get_phoneme(verbose=True)
    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)
    data = (X_train_scaled, X_test_scaled, y_train, y_test)

    ## Set up all models
    LIST_MODELS = []

    # Add machine learning models
    for seed in cf.LIST_SEEDS:
        svc = SVC(kernel="rbf", C=0.5, random_state=seed)
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", n_jobs=-1)
        dtc = DecisionTreeClassifier(max_depth=5, random_state=seed)
        rfc = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=seed)
        gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=5, random_state=seed)
        mlp = MLPClassifier(alpha=0.01, max_iter=300, hidden_layer_sizes=(15,), activation="relu", random_state=seed)
        LIST_MODELS.append([svc, seed, "SVM", "no_loss"])
        LIST_MODELS.append([knn, seed, "KNN", "no_loss"])
        LIST_MODELS.append([dtc, seed, "DT", "no_loss"])
        LIST_MODELS.append([rfc, seed, "RF", "no_loss"])
        LIST_MODELS.append([gbc, seed, "GB", "no_loss"])
        LIST_MODELS.append([mlp, seed, "MLP", "no_loss"])
    # Add traditional ANFIS
    for seed in cf.LIST_SEEDS:
        md1 = AnfisClassifier(num_rules=cf.DATA01['n_rules'], mf_class=cf.DATA01['mf_class'],
                              vanishing_strategy=cf.DATA01['vanishing_strategy'], act_output=None,
                              reg_lambda=None,
                              epochs=300, batch_size=128, optim="SGD", optim_params=None,
                              early_stopping=True, n_patience=50, epsilon=0.01, valid_rate=0.1,
                              seed=seed, verbose=True)
        LIST_MODELS.append([md1, seed, "ANFIS", "no_loss"])
    # Add GD ANFIS
    for opt in cf.LIST_GD_MODELS:
        for seed in cf.LIST_SEEDS:
            md2 = GdAnfisClassifier(num_rules=cf.DATA01['n_rules'], mf_class=cf.DATA01['mf_class'],
                                    vanishing_strategy=cf.DATA01['vanishing_strategy'], act_output="None",
                                    reg_lambda=None,
                                    epochs=300, batch_size=128, optim=opt["class"], optim_params=opt["paras"],
                                    early_stopping=True, n_patience=50, epsilon=0.01, valid_rate=0.1,
                                    seed=seed, verbose=True)
            LIST_MODELS.append([md2, seed, opt['name'], "no_loss"])
    # Add Bio ANFIS
    for opt in cf.LIST_BIO_MODELS:
        for seed in cf.LIST_SEEDS:
            md3 = BioAnfisClassifier(num_rules=cf.DATA01['n_rules'], mf_class=cf.DATA01['mf_class'],
                                     vanishing_strategy=cf.DATA01['vanishing_strategy'], act_output=None,
                                     reg_lambda=None,
                                     optim=opt["class"], optim_params=opt["paras"],
                                     obj_name=cf.OBJ_CLS, seed=seed, verbose=True)
            LIST_MODELS.append([md3, seed, opt['name'], "has_loss"])

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    ## Run parallel ==============================================================
    with ProcessPoolExecutor(max_workers=cf.N_WORKERS) as executor:
        futures = []
        for md_item in LIST_MODELS:
            futures.append(executor.submit(run_trial, md_item, data, cf))

        # Collect results as they complete
        for future in as_completed(futures):
            res_epoch_loss, res_predict = future.result()
            all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
            all_results.append(res_predict)  # Add evaluation result for this trial

    ## Run sequential =============================================================
    # for md_item in LIST_MODELS:
    #     res_epoch_loss, res_predict = run_trial(md_item, data, cf)
    #     all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
    #     all_results.append(res_predict)  # Add evaluation result for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{cf.PATH_SAVE}/{cf.DATA01['name']}/{cf.FILE_LOSS}", index=False, header=True)
    df_result.to_csv(f"{cf.PATH_SAVE}/{cf.DATA01['name']}/{cf.FILE_RESULT}", index=False, header=True)

    # Get metrics statistics
    stat_df = df_result.groupby("model_name").agg(["mean", "std"])
    stat_df.to_csv(f"{cf.PATH_SAVE}/{cf.DATA01['name']}/{cf.FILE_RESULT_STATS}", index=True, header=True)

    # Draw boxplot metrics, path_save, figsize=(8, 6), exts=(".png", ), verbose=False
    draw_boxplot(df_result, data_name=cf.DATA01['name'], list_models=cf.LIST_MODEL_CC,
                 metrics=cf.LIST_METRIC_CLS, path_save=f"{cf.PATH_SAVE}/{cf.DATA01['name']}/bp",
                 figsize=(8, 6), exts=(".png", ".pdf"), verbose=False)

    # Draw convergence chart
    draw_convergence_chart(df_loss, data_name=cf.DATA01['name'], list_models=cf.LIST_MODEL_CC,
                           path_save=f"{cf.PATH_SAVE}/{cf.DATA01['name']}/cc",
                           figsize=(8, 6), exts=(".png", ".pdf"), verbose=False)

    print(f"Done with data: {cf.DATA01['name']}")
