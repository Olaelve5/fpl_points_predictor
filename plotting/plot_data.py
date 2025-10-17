import matplotlib.pyplot as plt


def plot_data(df):
    """Plot EWMA points vs target score scatter plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df["ewma_points"], df["target_score"], alpha=0.5)
    plt.xlabel("EWMA Minutes")
    plt.ylabel("Target Score")
    plt.title("EWMA Minutes vs Target Score")
    plt.show()
