import time
from typing import Literal

from geopandas import read_file
from requests.exceptions import RequestException
from sqlalchemy import inspect

from utils.api_call import fetch_data
from utils.db import create_db_engine, init_db
from utils.helper import build_timeseries, points_in_boundaries
from utils.logger import setup_logger

MAX_RETRIES = 5
TABLE_NAME = "crowdedness"


def main(city: Literal["rotterdam", "amsterdam"], push: bool):
    BOUNDARIES = read_file(f"misc/{city}_.geojson")

    retries = 0

    with create_db_engine() as engine:
        if inspect(engine).has_table(TABLE_NAME):
            logging.info(f"Table {TABLE_NAME} already exists")
        else:
            init_db(engine, TABLE_NAME, push)
            logging.info(f"Created table {TABLE_NAME}")

        logging.info("Starting ETL Pipeline")
        with engine.connect() as conn:
            while True:
                try:
                    if (data := fetch_data()) is not None:
                        data = data.pipe(points_in_boundaries, BOUNDARIES).pipe(
                            build_timeseries
                        )
                        data.to_sql(
                            TABLE_NAME,
                            conn,
                            if_exists="append",
                            index=False,
                            method="multi",
                        )
                        n_rows = len(data)
                        logging.info(f"Pushed {n_rows} new rows | Waiting for new data")

                    elif retries <= MAX_RETRIES:
                        retries += 1
                        logging.info("No new data has been pulled. Waiting refresh")
                    else:
                        logging.warning(
                            f"No new data has been pulled for {MAX_RETRIES} minutes"
                        )

                    time.sleep(60)
                except RequestException as e:
                    logging.error(f"Request failed | {e}")
                except ValueError as e:
                    logging.error(f"Error parsing response | {e}")


if __name__ == "__main__":
    logging = setup_logger("etl_pipeline")
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--city",
        help="Select city for prediction",
        type=str,
        choices=["amsterdam", "rotterdam"],
        default="rotterdam",
    )
    parser.add_argument(
        "--insert",
        help="Choose whether you want to pre-load the db",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    kwargs = {k: v for k, v in args._get_kwargs() if v is not None}

    main(**kwargs)
