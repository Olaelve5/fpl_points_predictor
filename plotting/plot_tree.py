import matplotlib.pyplot as plt
import lightgbm as lgb


def plot_tree(model):
    """Plot a single decision tree from a LightGBM model."""
    lgb.plot_tree(model, figsize=(20, 10), show_info=["split_gain"])
    plt.title("LightGBM Decision Tree")
    plt.show()
