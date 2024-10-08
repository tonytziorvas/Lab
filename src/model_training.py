import os
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit

from utils import helper, logger


def main():
    df = helper.fetch_data().pipe(helper.pivot_table)
    df = df[[col for col in df.columns if df[col].var() > 100]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    lagged_df = helper.feature_extraction(df)

    target_columns = {}
    for target_district in df.columns[1:]:
        out, target_column = helper.create_crowd_levels(df, target_district)
        target_columns[target_column] = out

    target_labels = list(target_columns.keys())
    lagged_df = pd.concat(
        [
            lagged_df,
            pd.DataFrame(target_columns, columns=target_labels, index=lagged_df.index),
        ],
        axis=1,
    )

    lagged_df.set_index("timestamp", inplace=True)

    # TODO: store model variables in a config file
    models = [
        LGBMClassifier(
            boosting_type="gbdt",
            objective="multiclass",
            n_estimators=200,
            force_col_wise=True,
            num_class=3,
            num_leaves=3,
            max_depth=10,
            use_label_encoder=False,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
            device="gpu",
        )
        # XGBClassifier(
        #     objective="multi:softmax",
        #     n_estimators=200,
        #     num_class=3,
        #     gamma=0,
        #     n_jobs=-1,
        #     random_state=42,
        # ),
    ]

    cv = TimeSeriesSplit(n_splits=5)
    scoring = make_scorer(f1_score, average="micro")

    steps = [5, 15, 30, 60]
    n_targets = len(target_labels)

    root = Path("models")

    for district_label in target_labels:
        benchmarks = []

        district_name = district_label.split("_c_lvl")[0].replace(" ", "_")
        district_dir = root / district_name

        os.makedirs(district_dir, exist_ok=True)

        logging.info(f"Training model for {district_name}")
        for i, (model, step) in enumerate(product(models, steps)):
            model_name = model.__class__.__name__
            logging.info(f"Model: {model_name} | Prediction Step: {step}")

            X_train, X_test, y_train, y_test = helper.split_dataset(
                lagged_df,
                n_targets,
                district_label,
                step,
            )

            num_features = X_train.select_dtypes(include=np.number).columns.tolist()
            pipeline = helper.build_pipeline(model, num_features)
            # param_grid = {
            #     "classifier__n_estimators": np.arange(50, 300, 50),
            #     "classifier__max_depth": [3, 6, 10],
            #     "classifier__learning_rate": [0.01, 0.1, 0.3],
            #     "classifier__subsample": [0.5, 0.7, 1.0],
            #     "classifier__colsample_bytree": [0.5, 0.8, 1.0],
            #     "classifier__reg_alpha": [0.3, 0.7, 1.0],
            #     "classifier__reg_lambda": [0.3, 0.7, 1.0],
            # }
            try:
                # if i == 0:
                #     best_model = helper.grid_search(
                #         pipeline,
                #         param_grid,
                #         scoring,
                #         X_train,
                #         y_train,
                #         cv,
                #     )

                best_model = pipeline
                if best_model:
                    best_model.fit(X_train, y_train)
                    model_filename = f"{district_name}_{model_name}_{step}.joblib"
                    model_path = district_dir / model_filename

                    helper.evaluate(best_model, X_train, y_train, cv, scoring, "Train")
                    helper.evaluate(best_model, X_test, y_test, cv, scoring, "Test")

                    y_pred = best_model.predict(X_test)

                    joblib.dump(best_model, model_path)
                    logging.info(f"Model saved to {model_path}")

                    model_metrics = helper.log_metrics(
                        best_model, step, y_test, y_pred, model_path
                    )
                    benchmarks.append(model_metrics)

            except ValueError as e:
                logging.error(f"In model: {district_name} | {e}")
                os.remove
                continue
            i += 1

    if len(benchmarks) > 0:
        benchmarks = pd.concat(benchmarks, axis=0, ignore_index=True)
        benchmarks.to_json(
            f"reports/benchmarks/benchmark_{time.ctime()}.json",
            orient="records",
        )


if __name__ == "__main__":
    logging = logger.setup_logger("model_training")
    main()
