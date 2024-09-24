import logging
import os
import shutil
import tempfile
import time
import zipfile
from typing import Callable, Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.db import create_db_engine

TS_COL = "ts"


def fetch_data(
    chunk_size: int = 10**6,
    last_day_included: bool = False,
    last_api_call: bool = False,
) -> pd.DataFrame:
    table_name = "crowdedness"

    logging.info(f"Querying {table_name} table")

    with create_db_engine() as engine:
        with engine.connect() as conn:
            query = f"SELECT * FROM {table_name}"

            if last_day_included:
                yesterday = "CURRENT_TIMESTAMP::timestamp - INTERVAL '1 day'"
                yesterday_epoch = f"(SELECT EXTRACT(EPOCH FROM ({yesterday})))"
                query += f" where timestamp >= {yesterday_epoch};"
            elif last_api_call:
                max_ts = f"(SELECT MAX(timestamp) FROM {table_name})"
                query += f" WHERE timestamp = {max_ts};"

            chunks = pd.read_sql_query(sql=query, con=conn, chunksize=chunk_size)
        return pd.concat(chunks, ignore_index=True, axis=0)


def count_rows_in_csv(file_path):
    with open(file_path, "r") as file:
        row_count = sum(1 for _ in file) - 1
    return row_count


def points_in_boundaries(
    chunk: pd.DataFrame,
    city_boundaries: gpd.GeoDataFrame,
    ts_col: str = TS_COL,
) -> pd.DataFrame:
    """
    Process points in boundaries.

    Args:
        chunk (pd.DataFrame): Input data.
        city_boundaries (gpd.GeoDataFrame): City boundaries.
        ts_col (str): Timestamp column name. Defaults to TS_COL.

    Returns:
        gpd.GeoDataFrame: Processed data.
    """

    chunk = chunk.drop_duplicates().reset_index(drop=True)
    chunk = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"], crs=4326),
        crs=4326,
    )  # type: ignore

    df_left = pd.DataFrame(
        data=chunk.sindex.query(city_boundaries.geometry, predicate="intersects").T,
        columns=["district_id", "point_id"],
    ).reset_index(drop=True)

    df_right = (
        chunk.iloc[df_left["point_id"]][ts_col]
        .reset_index()
        .rename(columns={"index": "point_id", ts_col: "timestamp"})
    )

    merged = pd.merge(df_left, df_right, on="point_id")
    merged = merged.drop(columns=["name", "created_at", "geometry", "cartodb_id"])

    # Map district_id to district names
    district_codes = dict(city_boundaries.iloc[merged.district_id.unique()]["name"])
    merged["district_id"] = merged["district_id"].map(district_codes)

    return merged


def _process_csv(
    file: str,
    chunk_size: Optional[int] = 10**6,
    dtypes: Optional[Dict[str, str]] = None,
    filter_func: Optional[Callable] = None,
    city_boundaries: Optional[gpd.GeoDataFrame] = None,
    n_chunks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process a CSV file in chunks.

    Args:
        file (str): Path to the CSV file.
        chunk_size (int): Size of each chunk. Defaults to 10**6.
        dtypes (Optional[Dict[str, str]]): Data types of columns. Defaults to None.
        filter_func (Optional[Callable]): Function to filter chunks. Defaults to None.
        n_chunks (Optional[int]): Total number of chunks. Defaults to None.

    Returns:
        List[pd.DataFrame]: List of processed chunks.
    """

    all_chunks = []
    for chunk in tqdm(
        pd.read_csv(file, chunksize=chunk_size, dtype=dtypes),
        desc="Processing Chunks",
        total=n_chunks,
    ):
        chunk = filter_func(chunk, city_boundaries) if filter_func else chunk
        all_chunks.append(chunk)
    return pd.concat(all_chunks, ignore_index=True)


def process_csv_zip(
    zip_path: str,
    city_boundaries: gpd.GeoDataFrame,
    filter_func: Optional[Callable] = None,
    output_path: str = "output.parquet",
    chunk_size: int = 10**6,
    dtypes: Optional[Dict[str, str]] = None,
) -> None:

    logger = logging.getLogger("make_dataset")
    logger.info(f"Processing {zip_path}")
    temp_dir = tempfile.mkdtemp()

    try:
        logger.info(f"Unzipping {zip_path} to {temp_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_file = os.listdir(temp_dir)[0]
        logger.info(f"Processing {csv_file}")

        file_path = f"{temp_dir}/{csv_file}"
        n_rows = count_rows_in_csv(file_path)
        n_chunks = (n_rows // chunk_size) + 1

        df = _process_csv(
            file_path,
            chunk_size,
            dtypes,
            filter_func,
            city_boundaries,
            n_chunks,
        )

        logger.info(f"Saving to {output_path}")
        df.to_parquet(output_path, compression="gzip")
    finally:
        logger.info(f"Moving {zip_path} to data/raw/extracted")
        shutil.rmtree(temp_dir)
        shutil.move(zip_path, "data/raw/extracted")


def etl_pipeline(output_file, chunk_size, dtypes, city_boundaries, stream=False):
    logger = logging.getLogger("make_dataset")

    if not stream:
        # Load all csv files
        zip_paths = sorted(
            [
                os.path.join("data/raw", file)
                for file in os.listdir("data/raw")
                if file.endswith(".csv.zip")
            ]
        )

        # Process each csv file sequentially
        for zip_path in zip_paths:
            num_week = int(zip_path.split("_")[-1].split(".")[0])
            output_path = f"data/processed/points_per_district_week_{num_week}.parquet"
            process_csv_zip(
                zip_path,
                city_boundaries=city_boundaries,
                filter_func=points_in_boundaries,
                output_path=output_path,
                chunk_size=chunk_size,
                dtypes=dtypes,
            )

        # Merge all processed parquet files
        merge_processed_weeks(output_file)
    else:
        # Process each csv file sequentially
        csv_path = "src/data/raw/newdata.csv"

        df = pd.read_csv(csv_path, dtype=dtypes)
        df = points_in_boundaries(df, city_boundaries, ts_col="timestamp").reset_index(
            drop=True
        )

        df = build_timeseries(df)
        with db.create_db_engine() as engine:
            with engine.connect() as conn:
                logger.info("Connected to Database")

                i = 0
                while True:
                    if os.listdir("src/data/raw"):
                        df = pd.read_csv(csv_path, dtype=dtypes)
                        print(f"Shape: {df.shape}")
                        df = points_in_boundaries(
                            df, city_boundaries, ts_col="timestamp"
                        ).reset_index(drop=True)

                        df = build_timeseries(df)
                        print(f"Shape: {df.shape}")
                        df.to_sql(
                            "crowdedness",
                            conn,
                            if_exists="append",
                            index=False,
                            method="multi",
                        )
                        logger.info("Data pushed to database. Waiting for next batch")
                        shutil.move(csv_path, f"src/data/processed/newdata_{i}.csv")
                        conn.commit()
                        i += 1
                    else:
                        logger.warning(
                            "No new data has been pulled. Waiting for 60 seconds"
                        )

                    time.sleep(60)


def merge_processed_weeks(output_file):
    logger = logging.getLogger("make_dataset")

    weekly_data = [
        pd.read_parquet(f"data/processed/{week}")
        for week in os.listdir("data/processed")
        if week.startswith("points_")
    ]

    # Concat each dataframe and save to `final` folder
    logger.info("Concatenating dataframes")
    df = pd.concat(weekly_data, ignore_index=True)
    df = build_timeseries(df)

    df.to_parquet(f"{output_file}.parquet.gzip")
    logger.info("Dataset created")


def build_timeseries(df):
    return (
        df.groupby(by=["district_id", "timestamp"])
        .agg({"point_id": "count"})
        .rename({"point_id": "crowd"}, axis=1)
        .sort_values(by="timestamp")
        .reset_index()
    )


# Feature engineering
def feature_extraction(df, columns):
    logging.getLogger("model_training").info("Extracting temporal features")

    # Time-related features
    time_related_features = {
        "hour": df["timestamp"].dt.hour.astype(np.uint8),
        "day_of_week": df["timestamp"].dt.day_of_week.astype(np.uint8),
        "minute": df["timestamp"].dt.minute.astype(np.uint8),
        "is_weekend": (df["timestamp"].dt.weekday >= 5).astype(np.uint8),
    }  # 4 Features

    lagged_features = {}  # 19 Features per district
    rolling_features = {}  # len(windowds) * 5 Features per district
    exp_smoothing_features = {}  # len(windowds) Features per district
    windows = [5, 10, 15, 30] + [60 * i for i in range(1, 7)] + [60 * 24]
    lags = list(range(1, 11)) + [15, 30] + [60 * i for i in range(1, 7)] + [60 * 24]

    for district in columns:
        lagged_features |= {
            f"{district.replace(' ', '_')}_lag_{i}": df[district].shift(i).diff()
            for i in lags
        }

        rolling_features |= {
            f"{district.replace(' ', '_')}_rolling_{stat}_{window}": getattr(
                df[district].rolling(window=window), stat
            )()
            for window in windows
            for stat in ["mean", "std", "var", "skew", "kurt"]
        }

        exp_smoothing_features |= {
            f"{district.replace(' ', '_')}_ema_{window}": df[district]
            .ewm(span=window, adjust=True)
            .mean()
            for window in windows
        }

    return pd.concat(
        [
            df,
            pd.DataFrame(lagged_features),
            pd.DataFrame(rolling_features),
            pd.DataFrame(exp_smoothing_features),
            pd.DataFrame(time_related_features),
        ],
        axis=1,
    )


# TODO Add `n_bins` functionality
def create_crowd_levels(df, target_district):
    target_column = f'{target_district.replace(" ", "_")}_c_lvl'

    out = pd.qcut(
        df[target_district].rank(method="first"),
        q=[0, 0.3, 0.7, 1],
        labels=[0, 1, 2],
    )
    return out.astype(np.uint8), target_column


def split_dataset(lagged_df, n_targets, district, step):
    temp = lagged_df.copy(deep=True)
    temp[district] = temp[district].shift(-step)
    temp.dropna(inplace=True)
    temp[district] = temp[district].astype(np.uint8)

    X = temp[temp.columns[:-n_targets]]
    y = temp[district]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        shuffle=False,
    )

    return X_train, X_test, y_train, y_test


def grid_search(pipeline, param_grid, cv, scoring, X_train, y_train):
    logger = logging.getLogger("model_training")
    logger.info("Performing Grid Search")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        # refit="f1_micro",
        n_jobs=1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    logger.info("Grid Search Finished")

    print("\n====== Grid Search Results ======")
    print(f"Best score: {grid_search.best_score_:.3f}")
    print("Best parameters:\n ")
    for key, value in grid_search.best_params_.items():
        print(f"    - {key.split('__')[-1]}: {value}")

    for key, value in grid_search.cv_results_.items():
        if key.startswith(("mean_test", "test")):
            print(
                f"{key.replace('_', ' ').title()}: {value.mean():.3f} ± {value.std():.3f}"
            )

    return grid_search.best_estimator_


def build_pipeline(clf, num_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), num_features),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def evaluate(model, X, y, cv, scoring, set_name):
    print(f"\n====== {set_name} Set ======")
    cv_result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )
    for key, value in cv_result.items():
        if key.startswith(("mean_test", "test")):
            print(
                f"CV-{key.replace('_', ' ').title()}: {value.mean():.3f} ± {value.std():.3f}"
            )


def log_metrics(step, y_test, model, y_pred):
    model_name = model.__class__.__name__
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1_micro = f1_score(y_test, y_pred, average="micro")

    return pd.DataFrame(
        {
            "Model": [model_name],
            "Step": [step],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1_Micro": [f1_micro],
        }
    )


def pivot_table(df):
    df = (
        df.pivot_table(
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
    # df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df
