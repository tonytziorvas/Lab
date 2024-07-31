import os

import logger
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    Column,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
)

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
    return create_engine(connection_string, echo=False)


def create_schema(push=False):
    engine = make_connection()

    with engine.connect() as conn:
        logging.info("Connection established")
        try:
            meta = MetaData()

            crowd_table = Table(
                "crowdedness",
                meta,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("district_id", String(128), nullable=False),
                Column("timestamp", Integer, nullable=False),
                Column("crowd", Integer, nullable=False),
            )

            inspector = inspect(engine)
            if inspector.has_table(crowd_table.name):
                logging.info(f"Table {crowd_table.name} already exists")
            else:
                logging.info(f"Creating table {crowd_table.name}")
                crowd_table.create(engine)

                logging.info("Database Initialized")

            if push:
                push_data(conn)
        finally:
            conn.commit()
            engine.dispose()


def push_data(conn):
    logging.info("Pushing data")
    df = pd.read_parquet("data/final/points_per_district_full.parquet.gzip")
    df.to_sql(
        "crowdedness",
        con=conn,
        if_exists="append",
        index=False,
        method="multi",
    )

    logging.info("Data pushed to database")


if __name__ == "__main__":
    create_schema(push=True)
