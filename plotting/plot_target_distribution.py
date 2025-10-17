import matplotlib.pyplot as plt


def plot_target_distribution(y_test):
    """Plot histogram of target score distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=25, edgecolor="k", alpha=0.7)
    plt.xlabel("Target Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Target Scores")
    plt.show()
