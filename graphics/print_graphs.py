import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preparing.normalization.normalization_scallers import min_max_scaler, standard_scaler
from preparing.read_dataset import ReadDataset

def create_comparison_plot(data_original: pd.DataFrame, data_normalized: pd.DataFrame, x_col: str, y_col: str, title_original: str, title_normalized: str,) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].scatter(data_original[x_col], data_original[y_col], alpha=0.4, s=1)
    axes[0].set_title(title_original)
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)

    axes[1].scatter(data_normalized[x_col], data_normalized[y_col], alpha=0.4, s=1)
    axes[1].set_title(title_normalized)
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)

    plt.tight_layout()
    return fig

def main():
    X_train = ReadDataset("../datasets/result/train_test_split/training.parquet").read_data()
    X_test = ReadDataset("../datasets/result/train_test_split/testing.parquet").read_data()

    # Logarithm to reduce extreme values before normalization
    for col in ["Flow IAT Max", "Flow IAT Mean", "Total Fwd Packets"]:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    X_train_minmax, X_test_minmax = min_max_scaler(X_train, X_test)
    X_train_zscore, X_test_zscore = standard_scaler(X_train, X_test)

    # Min-Max graphs
    fig1 = create_comparison_plot(X_train, X_train_minmax, "Total Fwd Packets", "Flow IAT Max", "Original", "Min-Max")
    fig1.savefig("../graphics/min_max/flow_iat_max.png")

    fig2 = create_comparison_plot(X_train, X_train_minmax, "Total Fwd Packets", "Flow IAT Mean", "Original", "Min-Max")
    fig2.savefig("../graphics/min_max/flow_iat_mean.png")

    # Z-Score graphs
    fig3 = create_comparison_plot(X_train, X_train_zscore, "Total Fwd Packets", "Flow IAT Max", "Original", "Z-Score")
    fig3.savefig("../graphics/z_score/flow_iat_max.png")

    fig4 = create_comparison_plot(X_train, X_train_zscore, "Total Fwd Packets", "Flow IAT Mean", "Original", "Z-Score")
    fig4.savefig("../graphics/z_score/flow_iat_mean.png")

    plt.close("all")
    print("Graphs saved successfully")

if __name__ == "__main__":
    main()