import os
import pickle

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from utils import helper

STEPS = [5, 15, 30, 60]
BOUNDARIES = gpd.read_file("misc/rotterdam_.geojson")
MODEL_MAP = {"lgb": "LGBMClassifier", "xgb": "XGBClassifier"}
DATA_PATH = "data/processed/points_per_district_week_6.parquet"


def main():
    st.set_page_config(layout="wide")
    cols = st.columns([2, 1])

    # data = (
    #     pd.read_parquet(DATA_PATH)
    #     .pipe(helper.build_timeseries)
    #     .pipe(helper.pivot_table)
    # )
    data = helper.fetch_data().pipe(helper.pivot_table)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")

    missing_columns = set(BOUNDARIES.name) - set(data.columns)
    for col in missing_columns:
        data[col] = 0

    data = (
        helper.feature_extraction(data, data.columns[1:])
        .dropna()
        .reset_index(drop=True)
        .set_index("timestamp")
    )
    with st.sidebar:
        st.title("TrafficFlow Forecaster")
        st.subheader("Select Models and AOIs")

        selected_interval = st.selectbox(
            "Select Prediction Interval",
            options=STEPS,
            format_func=lambda x: f"{x} min",
            index=0,
        )

        model_name = st.selectbox(
            "Select Desired Model",
            options=["lgb", "xgb"],
            format_func=lambda x: MODEL_MAP.get(x),
        )

        st.divider()

        metrics = load_model_metrics()
        st.header(f"{MODEL_MAP.get(str(model_name))} Model Performance")
        st.markdown(
            """<style>.stProgress .st-bo {background-color: red;}</style>""",
            unsafe_allow_html=True,
        )

        for col in metrics.columns:
            col1, col2 = st.columns([1, 3])
            metric = metrics[col].values[0]
            col1.metric(col, f"{metric:.0%}")
            col2.progress(metric)

    models = load_models(str(model_name))
    last_record = data.index.max()
    predictions = predict_crowdedness(models, data.loc[data.index == last_record])
    predictions = predictions.rename(
        columns={5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min"}
    )

    with cols[0]:

        m = folium.Map(location=[51.9, 4.4], zoom_start=10)
        folium.Choropleth(
            geo_data=BOUNDARIES[["cartodb_id", "name", "geometry"]].to_json(),
            name="crowdedness",
            data=predictions,
            columns=["District", f"{selected_interval} min"],
            key_on="feature.properties.name",
            bins=[0, 1, 2, 3],
            fill_color="YlGnBu",
            nan_fill_color="pink",
            nan_fill_opacity=0.3,
            fill_opacity=0.8,
            line_weight=1.5,
            legend_name="Crowdedness",
        ).add_to(m)

        folium_static(m)

    with cols[1]:

        predictions = predictions.groupby(by="District").tail(1)
        st.dataframe(predictions, use_container_width=True, hide_index=True)


# Function to load the models based on the selection
@st.cache_data
def load_models(model_name: str = "lgb"):
    instance = MODEL_MAP.get(model_name, "LGBMClassifier")
    areas = sorted(BOUNDARIES.name.unique().tolist())

    models = {}
    for district in areas:
        FILE_PATH = f"models/{district}"

        if os.path.exists(FILE_PATH):
            area_models = {}
            for interval in STEPS:
                with open(f"{FILE_PATH}/{instance}_{interval}.pkl", "rb") as file:
                    model = pickle.load(file)
                    area_models[interval] = model

            models[district] = area_models

    return models


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Accuracy": [0.94, 0.87, 0.83, 0.64],
            "Precision": [0.92, 0.86, 0.83, 0.72],
            "Recall": [0.89, 0.82, 0.76, 0.58],
            "F1": [0.91, 0.85, 0.80, 0.7],
        }
    )


@st.cache_data
def predict_crowdedness(_models, data):
    # Make predictions
    predictions = []
    for model_name, model_list in _models.items():
        prediction = pd.DataFrame(
            data={
                interval: model.predict(data) for interval, model in model_list.items()
            },
            columns=model_list.keys(),
        )
        prediction["District"] = model_name
        predictions.append(prediction)

    return pd.concat(predictions, axis=0, ignore_index=True)


if __name__ == "__main__":
    main()
