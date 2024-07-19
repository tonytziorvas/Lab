import logging
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBClassifier

from utils import helper

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("reports/logs/model_training.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def main():
    logger.info("Starting model training...")

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

    logger.info("Feature engineering...")

    # 21 (initial) + 21*(12 + 4*5 + 4) + 4 (extracted)= 781 Features
    lagged_df = helper.feature_extraction(df, df.columns[1:]).reset_index(drop=True)

    # Build target label for all districts
    target_columns = {}
    for target_district in df.columns[1:]:
        labels, target_column = helper.create_crowd_levels(df, target_district)
        target_columns[target_column] = labels

    target_labels = [k for k in target_columns.keys()]
    lagged_df[target_labels] = pd.DataFrame(target_columns)

    lagged_df.set_index("timestamp", inplace=True)

    # TODO store model variables in a config file
    xgb = XGBClassifier(
        objective="multi:softmax",
        n_estimators=200,
        num_class=3,
        gamma=0,
        # colsample_bytree=0.7,
        # learning_rate=0.05,
        # max_depth=10,
        # reg_lambda=0.5,
        n_jobs=-1,
        random_state=42,
    )

    lgb = LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        n_estimators=200,
        force_col_wise=True,
        num_class=3,
        num_leaves=3,
        max_depth=10,
        use_label_encoder=False,
        # reg_lambda=0.7,
        # colsample_bytree=0.5,
        # learning_rate=0.01,
        # reg_alpha=0.3,
        # subsample=0.5,
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
    models = [xgb, lgb]

    n_targets = df.shape[1] - 1

    for district in target_labels:
        district_dir = f"models/{district.replace(' ', '_')}"
        os.makedirs(district_dir, exist_ok=True)
        logger.info(f"Model path: {district_dir}")

        for i, (model, step) in enumerate(product(models, n_steps)):
            model_name = model.__class__.__name__
            logger.info(f"Model: {model_name} | Prediction Step: {step}")

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

            num_features = X_train.select_dtypes(include=np.number).columns.tolist()

            pipeline = helper.build_pipeline(model, num_features)
            param_grid = {
                "classifier__n_estimators": np.arange(50, 300, 50),
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.3],
                "classifier__subsample": [0.5, 0.7, 1.0],
                "classifier__colsample_bytree": [0.5, 0.8, 1.0],
                "classifier__reg_alpha": np.linspace(0.3, 1.0, 3).round(1),
                "classifier__reg_lambda": np.linspace(0.3, 1.0, 3).round(1),
            }

            if i == 0:
                best_model = helper.grid_search(
                    pipeline,
                    param_grid,
                    ts_cv,
                    scoring,
                    X_train,
                    y_train,
                )

            best_model.fit(X_train, y_train)
            model_path = f"{district_dir}/{model_name}_{step}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
                logger.info(f"Model saved to {model_path}")

            helper.log_cv_results(best_model, X_train, y_train, ts_cv, scoring, "Train")
            helper.log_cv_results(best_model, X_test, y_test, ts_cv, scoring, "Test")

            y_pred = best_model.predict(X_test)

            helper.log_metrics(step, y_test, best_model, y_pred)


if __name__ == "__main__":
    main()
