from sqlalchemy import true
from models.advanced_points.def_expert import get_def_model
from models.advanced_points.fwd_expert import get_fwd_model
from models.advanced_points.mid_expert import get_mid_model
from models.advanced_points.gk_expert import get_gk_model
from utils.processing.get_train_test_data import get_train_test_data
import numpy as np
import joblib


class ExpertsModel:
    """
    Model that combines 4 expert models - one for each position.
    """

    def __init__(self, model_dir=None):
        if model_dir:
            self.load_models(model_dir)
        else:
            self.gk_model = get_gk_model()
            self.def_model = get_def_model()
            self.mid_model = get_mid_model()
            self.fwd_model = get_fwd_model()

            self.is_fitted = False

    def fit(self, x_train, y_train, x_test, y_test, save_dir=None):
        # Split training data on positions

        masks = {
            "gk": x_train["pos_GK"] == 1,
            "def": x_train["pos_DEF"] == 1,
            "mid": x_train["pos_MID"] == 1,
            "fwd": x_train["pos_FWD"] == 1,
        }

        test_masks = {
            "gk": (x_test["pos_GK"] == 1),
            "def": (x_test["pos_DEF"] == 1),
            "mid": x_test["pos_MID"] == 1,
            "fwd": x_test["pos_FWD"] == 1,
        }

        models = {
            "gk": self.gk_model,
            "def": self.def_model,
            "mid": self.mid_model,
            "fwd": self.fwd_model,
        }

        for pos, model in models.items():
            print(f"Training {pos.upper()} Expert...")

            # Filter Train
            X_tr_sub = x_train[masks[pos]]
            y_tr_sub = y_train[masks[pos]]

            # Filter Test
            X_te_sub = x_test[test_masks[pos]]
            y_te_sub = y_test[test_masks[pos]]

            if len(X_tr_sub) == 0:
                print(f"⚠️ Warning: No training data for {pos}")
                continue

            model.fit(
                X_tr_sub,
                y_tr_sub,
                eval_set=[(X_tr_sub, y_tr_sub), (X_te_sub, y_te_sub)],
                verbose=False,
            )

        self.is_fitted = True
        print("✅ All experts trained.")

        if save_dir:
            self.save_models(save_dir)

        return self

    def predict(self, x_test):
        predictions = np.zeros(len(x_test))

        # Goalkeepers
        mask_gk = x_test["pos_GK"] == 1
        if mask_gk.any():
            predictions[mask_gk] = self.gk_model.predict(x_test[mask_gk])

        # Defenders
        mask_def = x_test["pos_DEF"] == 1
        if mask_def.any():
            predictions[mask_def] = self.def_model.predict(x_test[mask_def])

        # Midfielders
        mask_mid = x_test["pos_MID"] == 1
        if mask_mid.any():
            predictions[mask_mid] = self.mid_model.predict(x_test[mask_mid])

        # Forwards
        mask_fwd = x_test["pos_FWD"] == 1
        if mask_fwd.any():
            predictions[mask_fwd] = self.fwd_model.predict(x_test[mask_fwd])

        return predictions

    def save_models(self, directory):
        print(f"Saving experts to {directory}...")
        joblib.dump(self.gk_model, f"{directory}/gk_model.pkl")
        joblib.dump(self.def_model, f"{directory}/def_model.pkl")
        joblib.dump(self.mid_model, f"{directory}/mid_model.pkl")
        joblib.dump(self.fwd_model, f"{directory}/fwd_model.pkl")
        print("✅ All position-specific models saved!")

    def load_models(self, directory):
        print(f"Loading experts from {directory}...")
        self.gk_model = joblib.load(f"{directory}/gk_model.pkl")
        self.def_model = joblib.load(f"{directory}/def_model.pkl")
        self.mid_model = joblib.load(f"{directory}/mid_model.pkl")
        self.fwd_model = joblib.load(f"{directory}/fwd_model.pkl")
        self.is_fitted = True
        print("✅ All position-specific models loaded!")


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, _ = get_train_test_data()

    model = ExpertsModel()
    model.fit(x_train, y_train, x_test, y_test, "data/saved_models/experts_model")

    predictions = model.predict(x_test)
    print(f"Made {len(predictions)} predictions")
