import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import helper

STEPS = [5, 15, 30, 60]

MODEL_MAP = {"lgb": "LGBMClassifier", "xgb": "XGBClassifier"}


def main():
    st.set_page_config(layout="wide")
    st.title("TrafficFlow Forecaster")

    cols = st.columns([2, 1])

    data, areas = prepare_data()

    with st.sidebar:
        area = st.selectbox("Select Area", sorted(areas.tolist()))
        model_name = st.selectbox("Select Model", ["lgb", "xgb"])

        models = load_models(model_name, str(area).replace(" ", "_"))

    with cols[0]:
        st.subheader("Crowdedness Forecast")

        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(8, 5))

        actuals, _ = helper.create_crowd_levels(data, area)
        predictions = predict_crowdedness(models, data)

        sns.lineplot(
            x=data.index[-200:],
            y=actuals.values[-200:],
            label="Actual Crowdedness",
            ax=ax,
        )
        alphas = np.arange(0.8, 0, -0.15)
        idxmax = actuals.index[-1]

        for prediction, alpha in zip(predictions, alphas):
            prediction = prediction.loc[prediction.index > idxmax]
            sns.lineplot(
                x=prediction.index,
                y=prediction[prediction.columns[0]],
                label=f"{prediction.columns[0]} min",
                linestyle="--",
                alpha=alpha,
                ax=ax,
            )

            idxmax = prediction.index[-1]

        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Crowdedness", fontsize=14)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([0, 1, 2], fontsize=10)
        ax.set_title(f"Crowdedness Forecast for {area}", fontsize=16)
        ax.legend(fontsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()

        sns.despine()
        st.pyplot(fig)

        data = load_model_metrics()

    with cols[1]:
        st.subheader(f"{MODEL_MAP.get(str(model_name))} Model Performance")
        st.markdown(
            """
            <style>
            .stProgress .st-bo {
                background-color: red;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        for col in data.columns:
            col1, col2 = st.columns([1, 3])
            col1.write(col)
            metric = data[col].values[0]
            col2.progress(metric, text=f"{metric:.0%}")


@st.cache_data
def load_model_metrics():

    # TODO Read `benchmarks.csv` and load metric for specified model
    metrics = pd.read_csv("reports/benchmarks.csv")
    metrics = pd.DataFrame(
        {
            "Accuracy": [0.94, 0.87, 0.83, 0.64],
            "Precision": [0.92, 0.86, 0.83, 0.72],
            "Recall": [0.89, 0.82, 0.76, 0.58],
            "F1": [0.91, 0.85, 0.80, 0.7],
        }
    )

    return {key: np.mean(value) for key, value in metrics.items()}


@st.cache_data
def prepare_data():
    city_boundaries = gpd.read_file("misc/rotterdam_.geojson")
    areas = city_boundaries.name.unique()

    data = pd.read_parquet("data/processed/points_per_district_week_4.parquet")
    data = data[data["timestamp"] > 1721013200]
    data = helper.points_in_boundaries(data, city_boundaries, "timestamp")
    data = helper.build_timeseries(data)
    data = helper.pivot_table(data)

    missing_columns = set(areas) - set(data.columns)
    for col in missing_columns:
        data[col] = 0

    data = (
        helper.feature_extraction(data, data.columns[1:])
        .dropna()
        .reset_index(drop=True)
        .set_index("timestamp")
    )

    return data, areas


# Function to load the model based on the selection
@st.cache_data
def load_models(model_name, district):
    instance = MODEL_MAP.get(model_name)

    return {
        step: pickle.load(open(f"models/{district}/{instance}_{step}.pkl", "rb"))
        for step in STEPS
    }


def predict_crowdedness(models, data):
    predictions = {step: model.predict(data) for step, model in models.items()}

    predictions = [
        pd.DataFrame(
            prediction,
            columns=[step],
            index=data.index + pd.Timedelta(minutes=step),
        )
        for step, prediction in predictions.items()
    ]

    return predictions


if __name__ == "__main__":
    main()
