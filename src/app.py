import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import tempdir
from typing import List

import folium
import geopandas as gpd
import joblib
import pandas as pd
import paramiko
import streamlit as st
from branca import colormap
from dotenv import load_dotenv
from streamlit_folium import folium_static

from utils import helper

STEPS = [5, 15, 30, 60]
MODEL_MAP = {"lgb": "LGBMClassifier", "xgb": "XGBClassifier"}
boundaries = gpd.read_file("misc/rotterdam_.geojson")
boundaries = boundaries[["cartodb_id", "name", "geometry"]]

load_dotenv("misc/.env")

# SSH and DB configuration from environment variables
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 22))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")


def main():
    st.set_page_config(layout="wide")
    cols = st.columns([2, 1])

    data = helper.fetch_data().pipe(helper.pivot_table)
    data = data[[col for col in data.columns if data[col].var() > 100]]
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
    areas = set(data.columns[1:])
    missing_columns = set(boundaries.name) - areas
    for col in missing_columns:
        data[col] = 0

    data = (
        helper.feature_extraction(data)
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

        for metric, value in metrics.items():
            col1, col2 = st.columns([1, 3])
            col1.metric(str(metric), f"{value:.0%}")
            col2.progress(value)

    models = load_models()

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


@st.cache_data
def load_models():
    models = {}

    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD)

        remote_path = "/ext1/models.zip"

        with ssh.open_sftp() as sftp:
            local_path = "models.zip"
            sftp.get(remote_path, local_path)

        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

                model_dir = Path(f"{temp_dir}/models")
                for district in model_dir.iterdir():
                    area_models = {}
                    for model in district.iterdir():
                        interval = model.name.split("_")[-1].split(".")[0]
                        area_models[interval] = joblib.load(model)

                    models[district.name] = area_models

        return models


@st.cache_data
def load_model_metrics():
    metrics_df = pd.read_json("reports/benchmarks/metrics.json")
    return metrics_df.mean(axis=0)


@st.cache_data
def predict_crowdedness(_models, data):
    # Make predictions
    predictions = []
    for model_name, model_list in _models.items():
        prediction = pd.DataFrame(
            data={interval: model.predict(data) for interval, model in model_list.items()},
            columns=model_list.keys(),
        )
        prediction["name"] = model_name
        predictions.append(prediction)

    merged = pd.concat(predictions, axis=0, ignore_index=True)
    merged = merged[["5", "15", "30", "60", "name"]]
    merged = merged.rename(
        columns={
            "5": "5 min",
            "15": "15 min",
            "30": "30 min",
            "60": "60 min",
        }
    )

    return merged


if __name__ == "__main__":
    main()
