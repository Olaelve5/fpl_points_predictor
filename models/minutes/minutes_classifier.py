import pandas as pd
from lightgbm import LGBMClassifier
from utils.get_training_test_data import get_train_test_data
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from models_operations.plot_model import plot_permutations


model = LGBMClassifier(
    n_estimators=457,
    learning_rate=0.01,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    subsample=0.63,
    max_depth=8,
    colsample_bytree=0.66,
)


def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)
    labels = ["Did Not Play", "Played"]

    # 3. Create the heatmap plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=labels, yticklabels=labels
    )

    # 4. Add titles and labels for clarity
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Player Playing Time")
    plt.show()


def plot_roc_curve(y_test, y_pred_proba):
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("ROC Curve for Player Playing Time Classifier")
    plt.show()


if __name__ == "__main__":
    training_data = get_train_test_data(minutes_training=True, minutes_classifier=True)
    X_train, X_test, y_train, y_test = training_data

    # --- THIS IS THE KEY CHANGE ---

    # 1. Calculate the ratio for scale_pos_weight
    neg_samples = y_train.value_counts()[0]
    pos_samples = y_train.value_counts()[1]
    scale_pos_weight_value = neg_samples / pos_samples

    print(f"Negative Samples: {neg_samples}, Positive Samples: {pos_samples}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

    # 2. Add the parameter to your model
    model = LGBMClassifier(
        n_estimators=457,
        learning_rate=0.01,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        subsample=0.63,
        max_depth=8,
        colsample_bytree=0.66,
        scale_pos_weight=scale_pos_weight_value,  # <-- Add the new parameter here
    )
    # ------------------------------------

    trained_model = model.fit(X_train, y_train)

    # Save the model to a file
    pd.to_pickle(trained_model, "data/saved_models/minutes_classifier_model.pkl")

    model_preds = trained_model.predict(X_test)
    model_proba = trained_model.predict_proba(X_test)[:, 1]

    print("Max classifier prediction:", model_proba.max())
    print("Min classifier prediction:", model_proba.min())

    plot_roc_curve(y_test, trained_model.predict_proba(X_test)[:, 1])
    plot_confusion_matrix(y_test, model_preds)
