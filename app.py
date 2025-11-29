import os
import re

import pandas as pd
import streamlit as st
import plotly.express as px

import main as hp_main  # reuse your training functions

# ---------- CONFIG ----------
TRAIN_PATH = os.path.join("data", "train.csv")
# ----------------------------


@st.cache_data
def load_data():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Could not find training data at {TRAIN_PATH}."
        )
    df = pd.read_csv(TRAIN_PATH)
    return df


# Human-friendly labels for numeric features
FEATURE_LABELS = {
    "MSSubClass": "Dwelling type (MSSubClass)",
    "LotFrontage": "Lot frontage (feet)",
    "LotArea": "Lot area (sq ft)",
    "OverallQual": "Overall material & finish quality",
    "OverallCond": "Overall condition",
    "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodeled",
    "MasVnrArea": "Masonry veneer area (sq ft)",
    "BsmtFinSF1": "Basement finished area 1 (sq ft)",
    "BsmtFinSF2": "Basement finished area 2 (sq ft)",
    "BsmtUnfSF": "Basement unfinished area (sq ft)",
    "TotalBsmtSF": "Total basement area (sq ft)",
    "1stFlrSF": "1st floor area (sq ft)",
    "2ndFlrSF": "2nd floor area (sq ft)",
    "LowQualFinSF": "Low quality finished area (sq ft)",
    "GrLivArea": "Above-ground living area (sq ft)",
    "BsmtFullBath": "Basement full baths",
    "BsmtHalfBath": "Basement half baths",
    "FullBath": "Full baths above grade",
    "HalfBath": "Half baths above grade",
    "BedroomAbvGr": "Bedrooms above grade",
    "KitchenAbvGr": "Kitchens above grade",
    "TotRmsAbvGrd": "Total rooms above grade",
    "Fireplaces": "Number of fireplaces",
    "GarageYrBlt": "Garage year built",
    "GarageCars": "Garage capacity (cars)",
    "GarageArea": "Garage area (sq ft)",
    "WoodDeckSF": "Wood deck area (sq ft)",
    "OpenPorchSF": "Open porch area (sq ft)",
    "EnclosedPorch": "Enclosed porch area (sq ft)",
    "3SsnPorch": "Three-season porch area (sq ft)",
    "ScreenPorch": "Screen porch area (sq ft)",
    "PoolArea": "Pool area (sq ft)",
    "MiscVal": "Miscellaneous value",
    "MoSold": "Month sold",
    "YrSold": "Year sold",
}


def pretty_feature_name(col: str) -> str:
    """Map raw column name to a nicer label."""
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]

    # Fallback: split CamelCase / digits and underscores
    s = col.replace("_", " ")
    s = re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", s)
    return s

def pretty_importance_name(raw: str) -> str:
    """
    Clean names like 'num__OverallQual' or 'cat__BsmtQual_Ex'
    into something readable.
    """
    # Strip the num__/cat__ prefix
    if "__" in raw:
        kind, rest = raw.split("__", 1)
    else:
        rest = raw

    # For categorical: something like 'BsmtQual_Ex'
    if rest.count("_") >= 1:
        col, cat = rest.split("_", 1)
        return f"{pretty_feature_name(col)} = {cat}"

    # For numeric: just the column
    return pretty_feature_name(rest)

@st.cache_resource
def compute_model_results():
    """Run the same training pipeline as main.py and return results."""
    df = hp_main.load_data(TRAIN_PATH)
    X, y = hp_main.split_features_target(df, target_col="SalePrice")
    preprocessor = hp_main.build_preprocessor(X)

    results_df = hp_main.evaluate_models(X, y, preprocessor, cv_splits=5)
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = hp_main.train_best_model(X, y, preprocessor, best_model_name)
    importance_df = hp_main.show_feature_importance(best_pipeline, X, top_k=15)

    return results_df, best_model_name, importance_df


def main():
    st.set_page_config(
        page_title="Housing Price Explorer",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Housing Price Explorer - Team 6 Project")

    st.markdown(
        """
        This app shows the dataset and model results for our **House Price Prediction**
        project.

        - Distribution of **SalePrice**  
        - Scatter plot of **SalePrice vs. any numeric feature**  
        - Model comparison table and top features from the training script
        """
    )

    df = load_data()

    results_df, best_model_name, importance_df = compute_model_results()

    st.markdown("## Model performance summary")

    if results_df is not None:
        st.dataframe(
            results_df.style.format(
                {
                    "rmse_mean": "{:,.0f}",
                    "rmse_std": "{:,.0f}",
                    "r2_mean": "{:.3f}",
                    "r2_std": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown(
            f"**Best model based on cross-validated RMSE:** `{best_model_name}`"
        )

    if importance_df is not None:
        st.markdown("### Top features for the best model")

        cleaned_importance = importance_df.copy()
        cleaned_importance["feature"] = cleaned_importance["feature"].apply(
            pretty_importance_name
        )

        st.dataframe(cleaned_importance, use_container_width=True)

    st.markdown("---")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("SalePrice", "Id")]

    label_to_col = {}
    for col in numeric_cols:
        label = pretty_feature_name(col)
        if label in label_to_col:
            label = f"{label} ({col})"
        label_to_col[label] = col

    st.markdown("## Relationship with SalePrice")
    feature_label = st.selectbox(
        "Choose a feature to compare with SalePrice:",
        options=sorted(label_to_col.keys()),
        index=0,
    )
    feature_col = label_to_col[feature_label]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sale Price Distribution")
        fig = px.histogram(df, x="SalePrice", nbins=50)
        fig.update_layout(xaxis_title="SalePrice", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader(f"SalePrice vs. {feature_label}")
        fig2 = px.scatter(
            df,
            x=feature_col,
            y="SalePrice",
            opacity=0.6,
        )
        fig2.update_layout(xaxis_title=feature_label, yaxis_title="SalePrice")
        st.plotly_chart(fig2, width="stretch")


if __name__ == "__main__":
    main()