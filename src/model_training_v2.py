import os
import shutil
from itertools import product

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from utils import db, helper, logger


def query_data(table, chunk_size=10**6):
    logging.info(f"Querying {table} table")
    with db.make_connection().begin() as connection:
        query = f"SELECT * FROM {table}"

        chunks = [pd.read_sql_query(sql=query, con=connection, chunksize=chunk_size)]
        return pd.concat(*chunks, ignore_index=True, axis=0)


def main():
    df = query_data("crowdedness")
    df = helper.pivot_table(df)

    lagged_df = helper.feature_extraction(df, df.columns[1:]).reset_index(drop=True)

    target_columns = {}
    logging.info("Encoding target labels")

    for target_district in df.columns[1:]:
        out, target_column = helper.create_crowd_levels(df, target_district)
        target_columns[target_column] = out

    target_labels = list(target_columns.keys())
    lagged_df[target_labels] = pd.DataFrame(target_columns)

    lagged_df.set_index("timestamp", inplace=True)

    # TODO store model variables in a config file
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
        ),
        XGBClassifier(
            objective="multi:softmax",
            n_estimators=200,
            num_class=3,
            gamma=0,
            n_jobs=-1,
            random_state=42,
        ),
    ]

    cv = TimeSeriesSplit(n_splits=5)
    scoring = make_scorer(f1_score, average="micro")

    n_steps = [5, 15, 30, 60]
    n_targets = len(target_labels)

    for district in target_labels:
        district_dir = f"models/{district.replace(' ', '_')}"
        logging.info(f"Training model for {district}")

        if os.path.exists(district_dir):
            logging.warning("Model exists. Skipping")
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

            param_grid = {
                # "classifier__n_estimators": np.arange(50, 300, 50),
                # "classifier__max_depth": [3, 6, 10],
                # "classifier__learning_rate": [0.01, 0.1, 0.3],
                "classifier__subsample": [0.5, 0.7, 1.0],
                # "classifier__colsample_bytree": [0.5, 0.8, 1.0],
                # "classifier__reg_alpha": np.linspace(0.3, 1.0, 3).round(1),
                # "classifier__reg_lambda": np.linspace(0.3, 1.0, 3).round(1),
            }
            try:
                if i == 0:

                    best_model = helper.grid_search(
                        pipeline,
                        param_grid,
                        cv,
                        scoring,
                        X_train,
                        y_train,
                    )
                best_model.fit(X_train, y_train)
                model_path = f"{district_dir}/{model_name}_{step}.pkl"

                with open(model_path, "wb") as f:
                    joblib.dump(best_model, f)
                    logging.info(f"Model saved to {model_path}")

                helper.evaluate_model(
                    best_model, X_train, y_train, cv, scoring, "Train"
                )
                helper.evaluate_model(best_model, X_test, y_test, cv, scoring, "Test")

                y_pred = best_model.predict(X_test)

                helper.log_metrics(step, y_test, best_model, y_pred)
            except ValueError as e:
                logging.error(f"In model: {model_name} | {e}")
                shutil.rmtree(district_dir)
                break
            i += 1


if __name__ == "__main__":
    logging = logger.setup_logger("model_training")
    main()
