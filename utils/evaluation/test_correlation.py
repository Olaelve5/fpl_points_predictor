import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_correlation(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    print(f"Features to consider dropping (Corr > 0.90): {to_drop}")

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    return to_drop


df = pd.read_csv("data/training_data/X_train.csv")
plot_correlation(df)
