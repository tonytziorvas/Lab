import os
import shutil
from itertools import product

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from utils import helper, logger


def main():
    df = helper.fetch_data().pipe(helper.pivot_table)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df[[col for col in df.columns if df[col].var() > 0]]

    lagged_df = helper.feature_extraction(df, df.columns[1:])

    target_columns = {}
    for target_district in df.columns[1:]:
        out, target_column = helper.create_crowd_levels(df, target_district)
        target_columns[target_column] = out

    target_labels = list(target_columns.keys())
    lagged_df[target_labels] = pd.DataFrame(target_columns)

    lagged_df.set_index("timestamp", inplace=True)

    # todo: store model variables in a config file
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

    cv = TimeSeriesSplit(n_splits=3)
    scoring = make_scorer(f1_score, average="micro")

    n_steps = [5, 15, 30, 60]
    n_targets = len(target_labels)

    for district in target_labels:
        benchmarks = []
        district_dir = f"models/{district.split('_c_lvl')[0]}"
        logging.info(f"Training model for {district_dir}")

        if os.path.exists(district_dir):
            logging.warning(f"Model {district_dir} exists. Skipping\n")
            continue
        else:
            os.makedirs(district_dir, exist_ok=True)

        for i, (model, step) in enumerate(product(models, n_steps)):
            model_name = model.__class__.__name__
            logging.info(f"Model: {model_name} | Prediction Step: {step}")

            X_train, X_test, y_train, y_test = helper.split_dataset(
                lagged_df,
                n_targets,
                district,
                step,
            )

            num_features = X_train.select_dtypes(include=np.number).columns.tolist()
            pipeline = helper.build_pipeline(model, num_features)
            # param_grid = {
            # "classifier__n_estimators": np.arange(50, 300, 50),
            # "classifier__max_depth": [3, 6, 10],
            # "classifier__learning_rate": [0.01, 0.1, 0.3],
            # "classifier__subsample": [0.5, 0.7, 1.0],
            # "classifier__colsample_bytree": [0.5, 0.8, 1.0],
            # "classifier__reg_alpha": np.linspace(0.3, 1.0, 3).round(1),
            # "classifier__reg_lambda": np.linspace(0.3, 1.0, 3).round(1),
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
                    model_path = f"{district_dir}/{model_name}_{step}.joblib"

                    helper.evaluate(best_model, X_train, y_train, cv, scoring, "Train")
                    helper.evaluate(best_model, X_test, y_test, cv, scoring, "Test")

                    y_pred = best_model.predict(X_test)

                    metrics = helper.log_metrics(step, y_test, best_model, y_pred)
                    benchmarks.append(metrics)

                    with open(model_path, "wb") as f:
                        joblib.dump(best_model, f)
                        logging.info(f"Model saved to {model_path}")

            except ValueError as e:
                logging.error(f"In model: {district_dir} | {e}")
                continue
            i += 1
        if len(benchmarks) > 0:
            benchmarks = pd.concat(benchmarks, axis=0, ignore_index=True)
            benchmarks.to_json("reports/benchmarks.json", orient="records")


if __name__ == "__main__":
    logging = logger.setup_logger("model_training")
    main()
