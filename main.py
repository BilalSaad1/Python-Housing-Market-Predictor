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

# ============================================================
# 1. DATA LOADING AND BASIC PREPARATION
# ============================================================
def load_data(train_path: str) -> pd.DataFrame:
    """Read the Kaggle training CSV and return the dataframe."""
    # Make sure the file exists before reading it
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found at {train_path}. "
            f"Make sure train.csv is downloaded from Kaggle."
        )

    df = pd.read_csv(train_path)
    print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "SalePrice"):
    """Split dataframe into feature matrix X and target vector y."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    cols_to_drop = [target_col]
    if "Id" in df.columns:
        cols_to_drop.append("Id")

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Split columns by data type
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Pipeline for numeric columns: fill missing values + scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Pipeline for categorical columns: fill missing values + one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Combine both pipelines so they are applied column-wise
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================
def get_models():
    """Return the dictionary of all models we compare in cross-validation."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
    # Add XGBoost only if it is available in the environment
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        )

    return models


# ============================================================
# 3. CROSS-VALIDATED EVALUATION (RMSE + R²)
# ============================================================
def evaluate_models(X, y, preprocessor, cv_splits: int = 5):
    """
    Run k-fold cross-validation for each model and return a summary DataFrame.

    Returned columns:
      model, rmse_mean, rmse_std, r2_mean, r2_std
    """
    models = get_models()
    results = []

    # KFold object defines how we split the data into train/validation folds
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        # Cross-validated RMSE (sklearn returns negative values for losses,
        # so we take the negative and turn it into a positive RMSE)
        rmse_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        # Cross-validated R² scores
        r2_scores = cross_val_score(
            pipe,
            X,
            y,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
        )

        # Aggregate metrics (mean and std across folds)
        rmse_mean = -rmse_scores.mean()
        rmse_std = rmse_scores.std()
        r2_mean = r2_scores.mean()
        r2_std = r2_scores.std()

        print(
            f"{name} - RMSE: {rmse_mean:.4f} (+/- {rmse_std:.4f}), "
            f"R^2: {r2_mean:.4f} (+/- {r2_std:.4f})"
        )

        results.append(
            {
                "model": name,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
            }
        )

    # Sort models by RMSE (best model at the top)
    results_df = pd.DataFrame(results).sort_values("rmse_mean")
    return results_df



# ============================================================
# 4. TRAIN BEST MODEL ON ALL TRAINING DATA
# ============================================================
def train_best_model(X, y, preprocessor, best_model_name: str):
    """Fit the best model on all training data and return the fitted pipeline."""
    models = get_models()
    if best_model_name not in models:
        raise ValueError(f"Unknown model name: {best_model_name}")

    best_model = models[best_model_name]
    # Full pipeline includes preprocessing and the chosen model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", best_model),
        ]
    )

    print(f"\nTraining best model on full data: {best_model_name}")
    pipeline.fit(X, y)

    # Evaluate performance on the entire training set (for reference)
    preds = pipeline.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y, preds)
    print(f"Train RMSE (on all data): {rmse:.4f}")
    print(f"Train R^2  (on all data): {r2:.4f}")

    return pipeline


# ============================================================
# 5. FEATURE IMPORTANCE INSPECTION
# ============================================================
def show_feature_importance(pipeline, X: pd.DataFrame, top_k: int = 15):
    """Return the top_k most important features for the fitted model."""
    print("\n=== Feature Importance / Coefficients ===")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Get the transformed feature names from the ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        print("Could not get feature names from preprocessor.")
        return None

    importances = None
    kind = None

     # Tree-based models: RandomForest, GradientBoosting, XGBoost
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        kind = "importance"
    # Linear models: LinearRegression, Ridge, Lasso
    elif hasattr(model, "coef_"):
        coef = model.coef_
        # If coef is 2D, collapse it to 1D
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        importances = np.abs(coef)
        kind = "|coefficient|"

    # If the model doesn't expose any importance information, stop here
    if importances is None:
        print(
            f"Model type {type(model)} does not provide feature importances or "
            "coefficients."
        )
        return None

    if len(importances) != len(feature_names):
        print("Mismatch between number of importances and feature names.")
        return None

    # Build a DataFrame of feature importances and keep only the top_k
    df_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                kind: importances,
            }
        )
        .sort_values(kind, ascending=False)
        .head(top_k)
    )

    print(df_importance)
    return df_importance


# ============================================================
# 6. KAGGLE TEST PREDICTION HELPER
# ============================================================
def predict_on_test(best_pipeline, test_path: str, output_path: str = "predictions.csv"):
    """Generate predictions on the Kaggle test set and save a submission CSV."""
    if not os.path.exists(test_path):
        print(f"\nTest file not found at {test_path}. Skipping test predictions.")
        return

    test_df = pd.read_csv(test_path)
    print(f"\nLoaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns.")

    # Kaggle expects 'Id' and 'SalePrice' columns in the submission
    test_ids = test_df["Id"]
    X_test = test_df.drop(columns=["Id"])

    preds = best_pipeline.predict(X_test)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    submission.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")


# ============================================================
# 7. MAIN ENTRY POINT – FULL TRAINING RUN
# ============================================================
def main():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    saved_model_path = "best_model.joblib"

    # Step 1–2: data loading and basic split
    df = load_data(train_path)
    X, y = split_features_target(df, target_col="SalePrice")
     # Step 3: build preprocessing
    preprocessor = build_preprocessor(X)

    # Step 4: evaluate all candidate models using k-fold CV
    results_df = evaluate_models(X, y, preprocessor, cv_splits=5)
    print("\n=== Model comparison (sorted by RMSE) ===")
    print(results_df.to_string(index=False))

    # Step 5: pick the best model by lowest RMSE
    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    print(f"\nBest model based on CV RMSE: {best_model_name}")

    # Step 6: train best model on the full dataset
    best_pipeline = train_best_model(X, y, preprocessor, best_model_name)
    # Step 7: feature importance
    show_feature_importance(best_pipeline, X, top_k=15)

    # Step 8: save the trained pipeline to disk for later reuse
    joblib.dump(best_pipeline, saved_model_path)
    print(f"\nSaved best model pipeline to {saved_model_path}")

    # Step 9: (optional) create predictions on test.csv
    predict_on_test(best_pipeline, test_path, output_path="kaggle_submission.csv")


if __name__ == "__main__":
    main()