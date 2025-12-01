import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import main as hp_main  # your main.py

# ---------- CONFIG ----------
TRAIN_PATH = os.path.join("data", "train.csv")
# ----------------------------


@st.cache_data
def load_data():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Could not find training data at {TRAIN_PATH}.")
    df = pd.read_csv(TRAIN_PATH)
    return df


# Nice human-readable labels for the main numeric features
FEATURE_LABELS = {
    "MSSubClass": "Dwelling type (MSSubClass)",
    "LotFrontage": "Lot frontage (ft)",
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

    # Fallback: split CamelCase / underscores
    s = col.replace("_", " ")
    s = re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", s)
    return s


def pretty_importance_name(raw: str) -> str:
    """
    Clean names like 'num__OverallQual' or 'cat__BsmtQual_Ex'
    into something readable.
    """
    if "__" in raw:
        _, rest = raw.split("__", 1)
    else:
        rest = raw

    if rest.count("_") >= 1:
        col, cat = rest.split("_", 1)
        return f"{pretty_feature_name(col)} = {cat}"

    return pretty_feature_name(rest)


@st.cache_resource
def compute_model_results():
    """Run the same training pipeline as main.py and return results + importance."""
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
        page_title="Housing Market Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Housing Market Predictor – Ames Dataset")

    st.markdown(
        """
        Interactive dashboard for our **House Price Prediction** project.

        - Explore the **SalePrice** distribution and log transform  
        - Inspect correlations between numeric features  
        - See **model comparison** results (RMSE / MAE / R²)  
        - View **top feature importances** from the best model
        """
    )

    df = load_data()
    results_df, best_model_name, importance_df = compute_model_results()

    # ------------------------------------------------------------------
    # DATASET EXPLORATION
    # ------------------------------------------------------------------
    st.header("Dataset exploration")

    st.subheader("SalePrice – before / after log scaling")

    col1, col2 = st.columns(2)

    with col1:
        fig_orig = px.histogram(df, x="SalePrice", nbins=50)
        fig_orig.update_layout(
            title="Original SalePrice",
            xaxis_title="SalePrice",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_orig, width="stretch")

    with col2:
        df_log = df.copy()
        df_log["SalePrice_log1p"] = np.log1p(df_log["SalePrice"])
        fig_log = px.histogram(df_log, x="SalePrice_log1p", nbins=50)
        fig_log.update_layout(
            title="Log-scaled SalePrice",
            xaxis_title="log(1 + SalePrice)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_log, width="stretch")

    st.markdown("### Correlation heatmap (top numeric features)")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("Id",)]

    corr_all = df[numeric_cols].corr()

    # IMPORTANT: use Series so sort_values works (fixes your error)
    target_corr = (
        corr_all.loc[:, "SalePrice"]
        .drop(labels=["SalePrice"])
        .abs()
        .sort_values(ascending=False)
    )

    top_corr_features = target_corr.head(10).index.tolist()
    heat_features = top_corr_features + ["SalePrice"]

    corr_subset = corr_all.loc[heat_features, heat_features]

    fig_heat = px.imshow(
        corr_subset,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        origin="lower",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, width="stretch")

    st.markdown("### SalePrice vs. selected numeric feature")

    # For the scatter, allow choosing any numeric feature (except Id & SalePrice)
    num_for_scatter = [c for c in numeric_cols if c != "SalePrice"]

    label_to_col = {}
    for col in num_for_scatter:
        label = pretty_feature_name(col)
        if label in label_to_col:
            label = f"{label} ({col})"
        label_to_col[label] = col

    feature_label = st.selectbox(
        "Choose a feature:",
        options=sorted(label_to_col.keys()),
        index=0,
    )
    feature_col = label_to_col[feature_label]

    fig_scatter = px.scatter(
        df,
        x=feature_col,
        y="SalePrice",
        opacity=0.6,
    )
    fig_scatter.update_layout(
        xaxis_title=feature_label,
        yaxis_title="SalePrice",
    )
    st.plotly_chart(fig_scatter, width="stretch")

    # ------------------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------------------
    st.header("Model evaluation")

    if results_df is not None:
        st.subheader("Cross-validated RMSE per model")

        # Bar chart with error bars (like your final report)
        fig_rmse = px.bar(
            results_df,
            x="model",
            y="rmse_mean",
            error_y="rmse_std",
            labels={
                "model": "Model",
                "rmse_mean": "CV RMSE (lower is better)",
            },
        )
        fig_rmse.update_layout(height=400)
        st.plotly_chart(fig_rmse, width="stretch")

        best_row = results_df[results_df["model"] == best_model_name].iloc[0]

        st.markdown(
            f"**Best model based on cross-validated RMSE:** "
            f"`{best_model_name}`"
        )

        st.subheader("Cross-validated metrics for best model")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{best_row['rmse_mean']:,.0f}")
        c2.metric("MAE", f"{best_row['mae_mean']:,.0f}")
        c3.metric("R²", f"{best_row['r2_mean']:.3f}")

        st.markdown("### Model performance table")
        st.dataframe(
            results_df.style.format(
                {
                    "rmse_mean": "{:,.0f}",
                    "rmse_std": "{:,.0f}",
                    "mae_mean": "{:,.0f}",
                    "mae_std": "{:,.0f}",
                    "r2_mean": "{:.3f}",
                    "r2_std": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCES
    # ------------------------------------------------------------------
    st.header("Top feature importances")

    if importance_df is not None:
        cleaned_importance = importance_df.copy()
        cleaned_importance["feature"] = cleaned_importance["feature"].apply(
            pretty_importance_name
        )

        value_col = [c for c in cleaned_importance.columns if c != "feature"][0]

        fig_imp = px.bar(
            cleaned_importance.sort_values(value_col, ascending=True),
            x=value_col,
            y="feature",
            orientation="h",
            labels={"feature": "Feature", value_col: "Importance"},
        )
        fig_imp.update_layout(height=500)
        st.plotly_chart(fig_imp, width="stretch")

        st.markdown("Top features table")
        st.dataframe(cleaned_importance, use_container_width=True)



if __name__ == "__main__":
    main()
