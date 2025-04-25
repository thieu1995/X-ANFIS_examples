#!/usr/bin/env python
# Created by "Thieu" at 02:07, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def draw_confusion_matrix(y_true, y_pred, figsize=(8, 6), title='Confusion Matrix',
                          pathsave="history/cm.png", verbose=False):
    label_fontsize = 15
    title_fontsize = 17
    annot_fontsize = 18

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", cbar=False,
                annot_kws={"size": annot_fontsize})  # Set annotation font size
    plt.xlabel('Predicted labels', fontsize=label_fontsize)
    plt.ylabel('True labels', fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)

    # Save the confusion matrix as an image (optional)
    Path("/".join(pathsave.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
    plt.savefig(pathsave, bbox_inches='tight')

    if verbose:
        plt.show()
    plt.close()



def draw_boxplot(df, data_name, list_models, metrics, path_save, figsize=(8, 6), exts=(".png", ), verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    # Select only the models in the list
    filtered_df = df[df['model_name'].isin(list_models)]

    # # Define a list of color palettes for each metric
    # color_palettes = ["Set2", "coolwarm", "Spectral", "cubehelix", "viridis", "Accent"]

    for metric in metrics:
        plt.figure(figsize=figsize)
        sns.boxplot(data=filtered_df, x="model_name", y=metric, palette="Spectral", hue="model_name")
        plt.title(f"Boxplot of {metric} metric on {data_name.capitalize()} dataset", fontsize=17)
        plt.xlabel("Models", fontsize=16)
        plt.ylabel(metric, fontsize=16)
        plt.xticks(rotation=45, fontsize=14, ha="right")
        plt.yticks(fontsize=14)

        for ext in exts:
            plt.savefig(f"{path_save}/bp_{metric}{ext}", bbox_inches="tight")
        if verbose:
            plt.show()
        plt.close()


def draw_convergence_chart(df, data_name, list_models, path_save, figsize=(8, 6), exts=(".png", ), verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    # Select only the models in the list
    filtered_df = df[df['model_name'].isin(list_models)]

    ## Draw convergence for each seed
    for seed in filtered_df['seed'].unique():
        plt.figure(figsize=figsize)
        # Lọc theo seed
        seed_df = filtered_df[filtered_df['seed'] == seed]
        # Duyệt qua từng model trong danh sách
        for model in list_models:
            model_df = seed_df[seed_df['model_name'] == model]
            plt.plot(model_df['epoch'], model_df['loss'], label=model)

        # Vẽ biểu đồ
        plt.title(f"Convergence chart of tested models on {data_name.capitalize()} dataset", fontsize=17)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # plt.grid(True)
        for ext in exts:
            plt.savefig(f"{path_save}/cc_{seed}{ext}", bbox_inches="tight")
        if verbose:
            plt.show()
        plt.close()

    ## Draw convergence mean
    # # Define a list of color palettes for each metric
    # color_palettes = ["Set2", "coolwarm", "Spectral", "cubehelix", "viridis", "Accent" ,"Dark2"]
    average_loss = filtered_df.groupby(["model_name", "epoch"])["loss"].mean().reset_index()

    # Plot the convergence chart
    plt.figure(figsize=figsize)
    sns.lineplot(data=average_loss, x="epoch", y="loss", hue="model_name", linewidth=2, palette="tab20")
    plt.title(f"Convergence chart of average loss on {data_name.capitalize()} dataset.", fontsize=17)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Average loss", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(title="Model", fontsize=12, title_fontsize=14)
    plt.legend(title="Model Name", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(alpha=0.3)

    for ext in exts:
        plt.savefig(f"{path_save}/cc-average{ext}", bbox_inches="tight")
    if verbose:
        plt.show()
    plt.close()
