import os
from typing import List

import folium
import geopandas as gpd
import joblib
import pandas as pd
import streamlit as st
from branca import colormap
from streamlit_folium import folium_static

from utils import helper

STEPS = [5, 15, 30, 60]
MODEL_MAP = {"lgb": "LGBMClassifier", "xgb": "XGBClassifier"}
boundaries = gpd.read_file("misc/rotterdam_.geojson")[
    ["cartodb_id", "name", "geometry"]
]


def main():
    st.set_page_config(layout="wide")
    cols = st.columns([2, 1])

    data = helper.fetch_data().pipe(helper.pivot_table)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
    areas = set(data.columns[1:])

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

    models = load_models(areas=list(areas), instance=str(model_name))
    last_record = data.index.max()
    predictions = predict_crowdedness(models, data.loc[data.index == last_record])

    with cols[0]:
        m = folium.Map(location=[51.9, 4.4], zoom_start=11)

        cm = colormap.StepColormap(
            ["gray", "green", "yellow", "red"],
            vmin=-1,
            vmax=3,
            index=[-1, 0, 1, 2],
            caption="Crowdedness Index",
        )
        merged = boundaries.merge(predictions, on="name", how="inner", sort=True)
        merged.fillna(-1, inplace=True)

        folium.GeoJson(
            merged.to_json(),
            style_function=lambda feature: {
                "fillColor": cm(feature["properties"][f"{selected_interval} min"]),
                "fillOpacity": 0.7,
                "color": "black",
                "weight": 1.5,
                "dashArray": "5, 5",
            },
            highlight_function=lambda x: {"fillOpacity": 0.9},
            tooltip=folium.GeoJsonTooltip(fields=["name"]),
        ).add_to(m)

        folium_static(m, width=800)
    with cols[1]:

        predictions = predictions.sort_values(by="name").groupby(by="name").tail(1)
        st.dataframe(
            predictions,
            hide_index=True,
            column_order=predictions["name"].unique().tolist().sort(),
            height=35 * len(predictions),
        )


# Function to load the models based on the selection
@st.cache_data
def load_models(areas: List[str], instance: str = "lgb"):
    instance = MODEL_MAP.get(instance, "LGBMClassifier")

    models = {}
    for district in areas:
        FILE_PATH = f"models/{district}"
        if os.path.exists(FILE_PATH):
            area_models = {}
            for interval in STEPS:
                with open(f"{FILE_PATH}/{instance}_{interval}.joblib", "rb") as file:
                    model = joblib.load(file)
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
        prediction["name"] = model_name
        predictions.append(prediction)

    merged = pd.concat(predictions, axis=0, ignore_index=True)
    merged = merged.rename(
        columns={
            5: "5 min",
            15: "15 min",
            30: "30 min",
            60: "60 min",
        }
    )

    return merged


if __name__ == "__main__":
    main()
