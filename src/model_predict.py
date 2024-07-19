import os
import pickle
import shutil
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from utils import helper


def main(model_name, district, city):
    instance = LGBMClassifier if model_name == "lgb" else XGBClassifier

    print(f"Loading {instance.__name__} Models... ")

    # Load model
    models = [
        pickle.load(open(f"models/{file}", "rb"))
        for file in os.listdir("models")
        if file.startswith(instance.__name__)
    ]

    # Check if new data has been pulled
    city_boundaries = gpd.read_file(f"misc/{city}_.geojson")
    i = 0

    while True:
        if os.listdir("src/data/raw"):

            df = build_timeseries(
                helper.points_in_boundaries(
                    chunk=pd.read_csv("src/data/raw/newdata.csv").drop_duplicates(),
                    city_boundaries=city_boundaries,
                    ts_col="timestamp",
                )
            )

            timestamps_needed = (
                pd.to_datetime(df["timestamp"].min(), unit="s") - pd.Timedelta(days=7)
            ).timestamp()

            df2 = pd.read_parquet("data/final/points_per_district_full.parquet.gzip")

            df2 = df2[df2["timestamp"] > timestamps_needed]

            points = pd.concat([df2, df], ignore_index=True).sort_values(by="timestamp")

            points = (
                points.pivot_table(
                    index="timestamp",
                    columns="district_id",
                    values="crowd",
                    aggfunc="sum",
                )
                .ffill()
                .bfill()
                .astype(np.uint16)
                .sort_values(by="timestamp")
                .reset_index()
            )
            for dis in city_boundaries.name.unique():
                if dis not in points.columns:
                    points[dis] = 0

            points = helper.feature_extraction(points, points.columns[1:]).drop(
                columns=district
            )

            # Make predictions
            predictions = []
            steps = [5, 15, 30, 60]

            for model, step in zip(models, steps):
                # thresh = pd.to_datetime(df["timestamp"].min(), unit="s")
                # temp = points[points.index > thresh]
                predictions.append(pd.DataFrame(model.predict(points), columns=[step]))

            df = pd.concat(predictions, axis=1).set_index(points.index)
            dest_path = f"src/data/predictions/predictions_{i}.json"
            with open(dest_path, "w") as f:
                df.astype(np.uint8).to_json(f)
                print(f"Data saved to {dest_path}")

            # Move data
            shutil.move(
                "src/data/raw/newdata.csv", f"src/data/processed/newdata_{i}.csv"
            )
            i += 1
            print("Waiting for next refresh...")
            time.sleep(60)
        else:
            print(f"[{time.ctime()}] - No data has been pulled.")
            time.sleep(30)


def build_timeseries(gdf):
    return (
        gdf.groupby(by=["district_id", "timestamp"])
        .agg({"point_id": "count"})
        .rename({"point_id": "crowd"}, axis=1)
        .sort_values(by="timestamp")
        .reset_index()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crowd Level Prediction API")
    parser.add_argument(
        "--model_name",
        help="Select Classifier model",
        choices=["lgb", "xgb"],
        type=str,
        default="lgb",
    )

    parser.add_argument(
        "--district",
        help="Select district for prediction",
        type=str,
        default="Rotterdam Centrum",
    )

    parser.add_argument(
        "--city",
        help="Select city for prediction",
        type=str,
        choices=["amsterdam", "rotterdam"],
        default="rotterdam",
    )

    args = parser.parse_args()
    model_name = args.model_name
    district = args.district
    city = args.city
    print("====== Initializing Prediction API ======")
    main(model_name, district, city)
