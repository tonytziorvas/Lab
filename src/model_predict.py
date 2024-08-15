import os
import pickle
import shutil
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from utils import helper, logger


def main(model_name, district, city):
    logging = logger.setup_logger("model_predict")
    logging.info("Starting prediction pipeline...")
    MODEL_MAP = {"lgb": "LGBMClassifier", "xgb": "XGBClassifier"}
    STEPS = [5, 15, 30, 60]
    BOUNDARIES = gpd.read_file(f"misc/{city}_.geojson")

    logging.info(f"Loading {MODEL_MAP[model_name]} Models")

    # Load model
    models = [
        pickle.load(open(f"models/{file}", "rb"))
        for file in os.listdir("models")
        if file.startswith(MODEL_MAP[model_name])  # type: ignore
    ]

    i = 0
    DATA_DIR = "data/raw"
    while True:
        df = helper.fetch_data(most_recent=True).pipe(helper.build_timeseries)

        # if os.listdir(f"{DATA_DIR}/raw"):
        #     df = helper.points_in_boundaries(
        #         chunk=pd.read_csv(f"{DATA_DIR}/raw/newdata.csv").drop_duplicates(),
        #         city_boundaries=BOUNDARIES,
        #         ts_col="timestamp",
        #     )
        #     df = helper.build_timeseries(df)

        timestamps_needed = (
            pd.to_datetime(df["timestamp"].min(), unit="s") - pd.Timedelta(days=7)
        ).timestamp()

        df2 = pd.read_parquet("data/final/points_per_district_full.parquet.gzip")
        df2 = df2[df2["timestamp"] > timestamps_needed]

        df = pd.concat([df2, df], ignore_index=True).sort_values(by="timestamp")
        df = helper.pivot_table(df)

        for dis in BOUNDARIES.name.unique():
            if dis not in df.columns:
                df[dis] = 0

        df = helper.feature_extraction(df, df.columns[1:]).drop(columns=district)

        predictions = [
            pd.DataFrame(model.predict(df), columns=[step])
            for model, step in zip(models, STEPS)
        ]
        df = pd.concat(predictions, axis=1).set_index(df.index)
        dest_path = f"{DATA_DIR}/predictions/predictions_{i}.json"
        with open(dest_path, "w") as f:
            df.astype(np.uint8).to_json(f)
            logging.info(f"Data saved to {dest_path}")

        shutil.move(
            f"{DATA_DIR}/raw/newdata.csv",
            f"{DATA_DIR}/processed/newdata_{i}.csv",
        )
        i += 1
        logging.info("Waiting for next refresh...")
        time.sleep(60)


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
    kwargs = {k: v for k, v in args._get_kwargs() if v is not None}

    main(**kwargs)
