import logging
import os
import shutil
import subprocess
import tempfile
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
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# logging.basicConfig(
#     format="[%(asctime)s] | %(message)s",
#     level=logging.INFO,
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("reports/logs/model_training.log"),
#     ],
# )
TS_COL = "ts"


def count_rows_in_csv(file_path):
    result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
    return int(result.stdout.split()[0]) - 1  # Subtract 1 for header row


def _process_csv(
    file: str,
    chunk_size: int = 10**6,
    dtypes: Optional[Dict[str, str]] = None,
    filter_func: Optional[Callable] = None,
    city_boundaries: Optional[gpd.GeoDataFrame] = None,
    n_chunks: Optional[int] = None,
) -> list[pd.DataFrame]:
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
    tqdm.pandas()
    all_chunks = []
    with pd.read_csv(file, chunksize=chunk_size, dtype=dtypes) as chunks:
        for chunk in tqdm(chunks, desc="Processing Chunks...", total=n_chunks):
            chunk = filter_func(chunk, city_boundaries) if filter_func else chunk
            all_chunks.append(chunk)
    return all_chunks


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
    processed_chunks = []

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
    merged = pd.merge(
        merged,
        chunk[["latitude", "longitude"]],
        left_on="point_id",
        right_index=True,
    )

    # Map district_id to district names
    district_codes = dict(city_boundaries.iloc[merged.district_id.unique()]["name"])
    merged["district_id"] = merged["district_id"].map(district_codes)
    processed_chunks.append(merged)
    return merged


def process_csv_zip(
    zip_path: str,
    filter_func: Optional[Callable] = None,
    city_boundaries: Optional[gpd.GeoDataFrame] = None,
    output_path: str = "output.parquet",
    chunk_size: int = 10**6,
    dtypes: Optional[Dict[str, str]] = None,
) -> None:

    temp_dir = tempfile.mkdtemp()
    try:
        logging.info(f"Unzipping {zip_path} to {temp_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for csv_file in os.listdir(temp_dir):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(temp_dir, csv_file)
                logging.info(f"Processing {csv_file}")
                n_rows = count_rows_in_csv(file_path)
                n_chunks = (n_rows // chunk_size) + 1
                chunks = _process_csv(
                    file_path,
                    chunk_size,
                    dtypes,
                    filter_func,
                    city_boundaries,
                    n_chunks,
                )
        processed_df = pd.concat(chunks)

        logging.info(f"Saving to {output_path}...")
        processed_df.to_parquet(output_path, compression="gzip")
    finally:
        shutil.rmtree(temp_dir)
        logging.info(f"Moving {zip_path} to data/raw/extracted")
        shutil.move(zip_path, "data/raw/extracted")


def etl_pipeline(output_file, chunk_size, dtypes, city_boundaries):
    current_week = len(os.listdir("data/processed")) + 1

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
        output_path = f"data/processed/points_per_district_week_{current_week}.parquet"

        process_csv_zip(
            zip_path,
            filter_func=points_in_boundaries,
            city_boundaries=city_boundaries,
            output_path=output_path,
            chunk_size=chunk_size,
            dtypes=dtypes,
        )

        current_week += 1

    # Merge all processed parquet files
    weekly_data = [
        pd.read_parquet(f"data/processed/{week}")
        for week in os.listdir("data/processed")
        if week.startswith("points_")
    ]

    # Concat each dataframe and save to `final` folder
    logging.info("Concatenating dataframes...")
    final_df = (
        pd.concat(weekly_data, ignore_index=True)
        .groupby(by=["district_id", "timestamp"])
        .agg({"point_id": "count"})
        .rename({"point_id": "crowd"}, axis=1)
        .sort_values(by="timestamp")
        .reset_index()
    )

    final_df.to_parquet(f"{output_file}.parquet.gzip")
    logging.info("Done!")


# Feature engineering
def feature_extraction(df, columns):

    # Time-related features
    time_related_features = {
        "hour": df["timestamp"].dt.hour.astype(np.uint8),
        "day_of_week": df["timestamp"].dt.day_of_week.astype(np.uint8),
        "minute": df["timestamp"].dt.minute.astype(np.uint8),
        "is_weekend": (df["timestamp"].dt.weekday >= 5).astype(np.uint8),
    }  # 4 Features

    lagged_features = {}  # 12 Features per district
    rolling_features = {}  # len(windowds) * 5 Features per district
    exp_smoothing_features = {}  # len(windowds) Features per district
    # windows = [5, 10, 15, 30] + [60 * i for i in range(1, 7)] + [60 * 24]
    # TODO - Restore the windows
    windows = [5, 10, 15, 30]

    for district in columns:
        lagged_features.update(
            {
                f"{district.replace(' ', '_')}_lag_{i}": df[district].shift(i).diff()
                for i in list(range(1, 11)) + [15, 30]
            }
        )

        rolling_features.update(
            {
                f"{district.replace(' ', '_')}_rolling_{stat}_{window}": getattr(
                    df[district].rolling(window=window), stat
                )()
                for window in windows
                for stat in ["mean", "std", "var", "skew", "kurt"]
            }
        )

        exp_smoothing_features.update(
            {
                f"{district.replace(' ', '_')}_ema_{window}": df[district]
                .ewm(span=window, adjust=True)
                .mean()
                for window in windows
            }
        )

    lagged_df = pd.concat(
        [
            df,
            pd.DataFrame(lagged_features),
            pd.DataFrame(rolling_features),
            pd.DataFrame(exp_smoothing_features),
            pd.DataFrame(time_related_features),
        ],
        axis=1,
    )

    return lagged_df


# TODO Add `n_bins` functionality
def create_crowd_levels(df, target_district):
    target_column = f'{target_district.replace(" ", "_")}_c_lvl'

    mean_crowd = df[target_district].mean()
    std_crowd = df[target_district].std()

    # Define bins based on mean and standard deviation
    # bins = [
    #     float("-inf"),
    #     mean_crowd - 1.0 * (std_crowd if std_crowd != 0 else 1),
    #     mean_crowd - 0.35 * (std_crowd if std_crowd != 0 else 1),
    #     mean_crowd + 0.35 * (std_crowd if std_crowd != 0 else 1),
    #     mean_crowd + 1.0 * (std_crowd if std_crowd != 0 else 1),
    #     float("inf"),
    # ]

    bins = [
        float("-inf"),
        mean_crowd - 0.55 * (std_crowd if std_crowd != 0 else 1),
        mean_crowd + 0.55 * (std_crowd if std_crowd != 0 else 1),
        float("inf"),
    ]
    out = pd.cut(
        df[target_district],
        bins=bins,
        labels=list(range(len(bins) - 1)),
        include_lowest=True,
    ).astype(np.uint8)

    return out, target_column


def grid_search(pipeline, param_grid, ts_cv, scoring, X_train, y_train):
    logging.info("Performing Grid Search")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=ts_cv,
        scoring=scoring,
        refit="f1_micro",
        n_jobs=1,
        verbose=1,
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)

    logging.info("Grid Search Finished")
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

    best_model = grid_search.best_estimator_

    return best_model


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


def log_cv_results(model, X, y, cv, scoring, set_name):
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


# TODO : Log the metrics in a .log file
def log_metrics(step, y_test, model, y_pred):
    model_name = model[1].__class__.__name__
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1_micro = f1_score(y_test, y_pred, average="micro")

    results = pd.DataFrame(
        {
            "Model": [model_name],
            "Step": [step],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1_Micro": [f1_micro],
        }
    )

    print(f"\nModel results:\n{results}\n")
