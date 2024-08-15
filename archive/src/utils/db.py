import os
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from psycopg2 import OperationalError
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
)
from sqlalchemy_utils import create_database, database_exists

from utils import logger

logging = logger.setup_logger("database")


def read_credentials():
    load_dotenv("misc/.env")

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")
    DB_NAME = os.getenv("DB_NAME")

    return DB_USER, DB_PASSWORD, HOST, PORT, DB_NAME


def make_connection(dialect: str = "psycopg2") -> Engine:
    logging.info("Loading Database Credentials")
    DB_USER, DB_PASSWORD, HOST, PORT, DB_NAME = read_credentials()
    connection_string = (
        f"postgresql+{dialect}://{DB_USER}:{DB_PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
    )

    logging.info(f"Connecting to {DB_NAME} DB on {HOST}:{PORT}")
    engine = create_engine(
        connection_string,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    if database_exists(engine.url):
        logging.info(f"Database {engine.url.database} exists")
    else:
        create_database(engine.url)
        logging.info(f"Database {engine.url.database} created")

    return engine


def init_db(push: bool = True) -> Literal[-1, 0, 1]:
    engine = make_connection()
    table_name = "crowdedness"

    try:
        with engine.connect() as conn:
            logging.info("Connection established")

            if inspect(engine).has_table(table_name):
                logging.info(f"Table {table_name} already exists")
            else:
                create_table(engine, table_name)
                logging.info(f"Created table {table_name}")

            if push:
                push_data(conn)
                return 1
            return 0
    except OperationalError as e:
        logging.error(f"Connection failed | {e}")
        return -1


def create_table(engine: Engine, table_name: str):
    meta = MetaData()
    crowd_table = Table(
        table_name,
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("district_id", String(128), nullable=False),
        Column("timestamp", Integer, nullable=False),
        Column("crowd", Integer, nullable=False),
    )

    crowd_table.create(engine)


def push_data(
    conn: Connection,
    file_path: str = "data/final/points_per_district_full.parquet.gzip",
    table_name: str = "crowdedness",
):
    logging.info("Pushing data")
    pd.read_parquet(file_path).to_sql(
        table_name,
        con=conn,
        if_exists="append",
        index=False,
        method="multi",
    )

    logging.info("Data pushed to database")


if __name__ == "__main__":
    init_db()
