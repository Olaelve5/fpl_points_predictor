import matplotlib.pyplot as plt


def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted values with a diagonal reference line."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Target Score")
    plt.ylabel("Predicted Target Score")
    plt.title("Actual vs Predicted Target Score")
    plt.show()

    # Save the plot
    plt.savefig("data/saved_plots/preds_vs_actual.png")
