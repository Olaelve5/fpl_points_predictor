import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from utils.processing.get_train_test_data import get_train_test_data
import xgboost as xgb
from models.points.gradient_boost.model import model_params
from models.points.gradient_boost.get_columns_to_drop import get_columns_to_drop


def train_voting_model(X_train, y_train, weights=[0.3, 0.1, 0.6]):
    # Expert A: XGBoost
    xgb_params = model_params.copy()
    xgb_params.pop("early_stopping_rounds", None)
    xgb_model = XGBRegressor(**xgb_params)

    # Expert B: Random Forest
    rf_model = xgb.XGBRFRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    )

    # Expert C: Ridge (Wrapped in Imputer)
    linear_model = make_pipeline(SimpleImputer(strategy="mean"), Ridge(alpha=1.0))

    # Tune voting weights
    ensemble = VotingRegressor(
        estimators=[("xgb", xgb_model), ("rf", rf_model), ("linear", linear_model)],
        weights=weights,
        n_jobs=-1,
    )

    print("Training Voting Ensemble...")

    trained_model = ensemble.fit(X_train, y_train)

    return trained_model


if __name__ == "__main__":
    X_train, _, y_train, _, _ = get_train_test_data(test_season="25_26")

    cols_to_drop = get_columns_to_drop()
    X_train = X_train.drop(columns=cols_to_drop, errors="ignore")

    trained_model = train_voting_model(X_train, y_train)

    joblib.dump(trained_model, "data/saved_models/points/voting_model.pkl")
    print("✅ Voting Model Saved!")
