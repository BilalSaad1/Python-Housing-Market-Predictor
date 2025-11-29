import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed; skipping XGBRegressor.")


RANDOM_STATE = 42


def load_data(train_path: str) -> pd.DataFrame:
    """Load the Kaggle House Prices training data."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}. "
                                f"Make sure train.csv is downloaded from Kaggle.")

    df = pd.read_csv(train_path)
    print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "SalePrice"):
    """Separate features X and target y, and drop Id if present."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    cols_to_drop = [target_col]
    if "Id" in df.columns:
        cols_to_drop.append("Id")

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def get_models():
    """Define the models you mentioned in your proposal."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE
        )

    return models


def evaluate_models(X, y, preprocessor, cv_splits: int = 5):
    """Run cross-validation for each model and return a summary DataFrame."""
    models = get_models()
    results = []

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        rmse_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        r2_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=kf,
            scoring="r2",
            n_jobs=-1
        )

        rmse_mean = -rmse_scores.mean()
        rmse_std = rmse_scores.std()
        r2_mean = r2_scores.mean()
        r2_std = r2_scores.std()

        print(f"{name} - RMSE: {rmse_mean:.4f} (+/- {rmse_std:.4f}), "
              f"R^2: {r2_mean:.4f} (+/- {r2_std:.4f})")

        results.append({
            "model": name,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std
        })

    results_df = pd.DataFrame(results).sort_values("rmse_mean")
    return results_df


def train_best_model(X, y, preprocessor, best_model_name: str):
    """Fit the best model on the full dataset and return the fitted pipeline."""
    models = get_models()
    if best_model_name not in models:
        raise ValueError(f"Unknown model name: {best_model_name}")

    best_model = models[best_model_name]
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    print(f"\nTraining best model on full data: {best_model_name}")
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    mse = mean_squared_error(y, preds)  
    rmse = mse ** 0.5                   
    r2 = r2_score(y, preds)
    print(f"Train RMSE (on all data): {rmse:.4f}")
    print(f"Train R^2 (on all data): {r2:.4f}")

    return pipeline


def show_feature_importance(pipeline, X: pd.DataFrame, top_k: int = 15):
    """Return top_k features by importance for tree-based or linear models."""
    print("\n=== Feature Importance / Coefficients ===")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        print("Could not get feature names from preprocessor.")
        return None

    importances = None
    kind = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "importance"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        importances = np.abs(coef)
        kind = "|coefficient|"

    if importances is None:
        print(f"Model type {type(model)} does not provide feature importances or coefficients.")
        return None

    if len(importances) != len(feature_names):
        print("Mismatch between number of importances and feature names.")
        return None

    df_importance = pd.DataFrame({
        "feature": feature_names,
        kind: importances
    }).sort_values(kind, ascending=False).head(top_k)

    print(df_importance)
    return df_importance


def predict_on_test(best_pipeline, test_path: str, output_path: str = "predictions.csv"):
    """Make predictions on Kaggle test.csv and save to a CSV for submission (optional)."""
    if not os.path.exists(test_path):
        print(f"\nTest file not found at {test_path}. Skipping test predictions.")
        return

    test_df = pd.read_csv(test_path)
    print(f"\nLoaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")

    test_ids = test_df["Id"]
    X_test = test_df.drop(columns=["Id"])

    preds = best_pipeline.predict(X_test)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")


def main():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")  
    saved_model_path = "best_model.joblib"

    df = load_data(train_path)

    X, y = split_features_target(df, target_col="SalePrice")

    preprocessor = build_preprocessor(X)

    results_df = evaluate_models(X, y, preprocessor, cv_splits=5)
    print("\n=== Model comparison (sorted by RMSE) ===")
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    print(f"\nBest model based on CV RMSE: {best_model_name}")

    best_pipeline = train_best_model(X, y, preprocessor, best_model_name)

    show_feature_importance(best_pipeline, X, top_k=15)

    joblib.dump(best_pipeline, saved_model_path)
    print(f"\nSaved best model pipeline to {saved_model_path}")

    predict_on_test(best_pipeline, test_path, output_path="kaggle_submission.csv")


if __name__ == "__main__":
    main()