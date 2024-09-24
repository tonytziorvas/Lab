import logging
import pickle
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from utils import helper

logging.basicConfig(
    format="[%(asctime)s] | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reports/model_training.log"),
    ],
)


def main():
    logging.info("Starting model training...")

    df = pd.read_parquet("data/final/points_per_district_full.parquet.gzip")
    df = (
        df.pivot_table(
            index="timestamp", columns="district_id", values="crowd", aggfunc="sum"
        )
        .ffill()
        .bfill()
        .astype(np.uint16)
        .sort_values(by="timestamp")
        .reset_index()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    print(
        "Data gathered from "
        f"{pd.to_datetime(df['timestamp'].min(), unit='s')} to "
        f"{pd.to_datetime(df['timestamp'].max(), unit='s')}"
    )

    logging.info("Feature engineering...")

    # 21 (initial) + 21*(12 + 4*5 + 4) + 4 (extracted)= 781 Features
    lagged_df = helper.feature_extraction(df, df.columns[1:]).reset_index()

    # Build target label for the district
    target_district = "Rotterdam Centrum"
    labels, target_column = helper.create_crowd_levels(df, target_district)

    lagged_df[target_column] = labels
    lagged_df.drop(columns=target_district, inplace=True)

    # xgb = XGBClassifier(
    #     objective="multi:softmax",
    #     n_estimators=200,
    #     num_class=3,
    #     gamma=0,
    #     # colsample_bytree=0.7,
    #     # learning_rate=0.05,
    #     # max_depth=10,
    #     # reg_lambda=0.5,
    #     n_jobs=-1,
    #     random_state=42,
    # )

    lgb = LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        n_estimators=200,
        force_col_wise=True,
        num_class=3,
        num_leaves=3,
        max_depth=10,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    )

    ts_cv = TimeSeriesSplit(n_splits=5)
    scoring = {
        "accuracy": make_scorer(balanced_accuracy_score),
        "f1_micro": make_scorer(f1_score, average="micro"),
    }

    n_steps = [5, 15, 30, 60]

    models = [lgb]
    for model, step in product(models, n_steps):
        model_name = model.__class__.__name__
        print(f"====== Model: {model_name} --- Prediction Step: {step} ======")

        temp = lagged_df.copy(deep=True)

        shifted = temp[target_column].shift(-step)
        temp[target_column] = shifted

        temp = temp.dropna().reset_index(drop=True).set_index("timestamp")
        temp[target_column] = temp[target_column].astype(np.uint8)

        X, y = temp.drop(columns=target_column), temp[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            shuffle=False,
        )

        num_features = X_train.select_dtypes(include=np.number).columns.tolist()
        logging.info(f"No. of features used: {len(num_features)}")

        pipeline = helper.build_pipeline(model, num_features)

        param_grid = {
            # "classifier__estimator__n_estimators": np.arange(50, 300, 50),
            "classifier__estimator__max_depth": [3, 6, 10],
            "classifier__estimator__learning_rate": [0.01, 0.1, 0.3],
            # "classifier__estimator__subsample": [0.5, 0.7, 1.0],
            # "classifier__estimator__colsample_bytree": [0.5, 0.8, 1.0],
            # "classifier__estimator__reg_alpha": np.linspace(0.3, 1.0, 3).round(1),
            # "classifier__estimator__reg_lambda": np.linspace(0.3, 1.0, 3).round(1),
        }

        logging.info(
            f"Building pipeline for {model_name} with prediction step {step}..."
        )

        best_model = helper.grid_search(
            pipeline,
            param_grid,
            ts_cv,
            scoring,
            X_train,
            y_train,
        )

        best_model.fit(X_train, y_train)
        model_path = f"models/{model.__class__.__name__}_{step}_v2.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
            logging.info(f"Model saved to {model_path}")

        helper.evaluate(best_model, X_train, y_train, ts_cv, scoring, "Train")
        helper.evaluate(best_model, X_test, y_test, ts_cv, scoring, "Test")

        y_pred = best_model.predict(X_test)

        helper.log_metrics(step, y_test, best_model, y_pred)


if __name__ == "__main__":
    main()
