import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def points_in_boundaries(
    chunk: pd.DataFrame, city_boundaries: gpd.GeoDataFrame, ts_col: str = TS_COL
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


def build_timeseries(df):
    return (
        df.groupby(by=["district_id", "timestamp"])
        .agg({"point_id": "count"})
        .rename({"point_id": "crowd"}, axis=1)
        .sort_values(by="timestamp")
        .reset_index()
    )


# Feature engineering
def feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("model_training")
    logger.info("Extracting temporal features")

    # Time-related features
    time_related_features = {
        "hour": df["timestamp"].dt.hour.astype(np.uint8),
        "day_of_week": df["timestamp"].dt.day_of_week.astype(np.uint8),
        "minute": df["timestamp"].dt.minute.astype(np.uint8),
        "is_weekend": (df["timestamp"].dt.weekday >= 5).astype(np.uint8),
    }  # 4 Features

    lagged_features = {}  # 19 Features per district
    rolling_features = {}  # len(windows) * 5 Features per district
    exp_smoothing_features = {}  # len(windows) Features per district
    windows = [5, 10, 15, 30] + [60 * i for i in range(1, 7)] + [60 * 24]
    lags = list(range(1, 11)) + [15, 30] + [60 * i for i in range(1, 7)] + [60 * 24]

    for district in df.columns[1:]:
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
def create_crowd_levels(df: pd.DataFrame, target_district: str) -> Tuple[pd.Series, str]:
    target_column = f'{target_district.replace("_", " ")}_c_lvl'

    out = pd.qcut(
        df[target_district].rank(method="first"),
        q=[0, 0.3, 0.7, 1],  # q=[0, 0.2, 0.45, 0.65, 0.8, 1] for 5 bins
        labels=[0, 1, 2],
    )
    return out.astype(np.uint8), target_column


def split_dataset(lagged_df, n_targets, district, step):
    temp = lagged_df.copy(deep=True)
    temp[district] = temp[district].shift(-step)
    temp = temp.dropna()
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


# TODO: Replace GridSearchCV with Bayesian Search
def grid_search(pipeline, param_grid, scoring, X_train, y_train, cv):
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
    try:
        grid_search.fit(X_train, y_train)
        logger.info("Grid Search Finished")

        print("\n====== Grid Search Results ======")
        print(f"Best score: {grid_search.best_score_:.3f}")
        print("Best parameters:\n ")
        for key, value in grid_search.best_params_.items():
            print(f"    - {key.split('__')[-1]}: {value}")

        for key, value in grid_search.cv_results_.items():
            if key.startswith(("mean_test", "test")):
                key = key.replace("_", " ").title()
                print(f"{key}: {value.mean():.3f} ± {value.std():.3f}")

        return grid_search.best_estimator_
    except Exception as e:
        print("\n --- ERROR in Grid Search - SKIP ---\n")
        print(e)
        return


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
    cv_result = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score="raise")

    for key, value in cv_result.items():
        if key.startswith(("mean_test", "test")):
            metric = f"CV-{key.replace('_', ' ').title()}: {value.mean():.3f} ± {value.std():.3f}"

    print("+" + "-" * (30 + 2) + "+")
    print("| " + f"{set_name} Set".center(30) + " |")
    print("| " + metric.center(30) + " |")
    print("+" + "-" * (30 + 2) + "+")


def log_metrics(model, step, y_test, y_pred, model_path) -> Dict:

    model_name = model.__class__.__name__
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1_micro = f1_score(y_test, y_pred, average="micro")

    model_size = os.stat(model_path).st_size / 1024
    # params = model.get_params()["classifier"].get_params()

    row_dict = {
        "Model": model_name,
        "Step": step,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Micro": f1_micro,
        "Size": model_size,
    }
    # row_dict |= params

    return row_dict


def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
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
