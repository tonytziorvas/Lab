import os
from contextlib import contextmanager
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
)
from sqlalchemy.exc import SQLAlchemyError
from sshtunnel import SSHTunnelForwarder

from utils.logger import setup_logger

load_dotenv("misc/.env")

# SSH and DB configuration from environment variables
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 22))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

logger = setup_logger("database")


@contextmanager
def __create_ssh_tunnel():

    tunnel = None
    try:
        logger.info("Setting up SSH tunnel")
        tunnel = SSHTunnelForwarder(
            (SSH_HOST, SSH_PORT),
            ssh_username=SSH_USER,
            ssh_password=SSH_PASSWORD,
            remote_bind_address=(DB_HOST, DB_PORT),
            local_bind_address=("127.0.0.1", 5432),
        )
        tunnel.start()
        logger.info(f"SSH tunnel established on remote port {tunnel.local_bind_port}.")
        yield tunnel
    except Exception as e:
        logger.error(f"Failed to establish SSH tunnel: {e}")
        raise
    finally:
        if tunnel:
            tunnel.stop()
            logger.info("SSH tunnel closed.")


@contextmanager
def create_db_engine():
    """
    Creates an SQLAlchemy engine connected through the SSH tunnel.
    Yields the engine for use in a with statement.
    """
    with __create_ssh_tunnel() as tunnel:
        engine = None
        try:
            logger.info("Creating SQLAlchemy engine")
            connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{tunnel.local_bind_port}/{DB_NAME}"

            engine = create_engine(
                connection_string, echo=False, pool_pre_ping=True, pool_recycle=3600
            )
            yield engine
        except SQLAlchemyError as e:
            logger.error(f"Failed to create engine or connect to database | {e}")
            raise
        finally:
            if engine:
                engine.dispose()
                logger.info("SQLAlchemy engine disposed")


def init_db(engine: Engine, table_name: str, push: bool = True) -> Literal[-1, 0, 1]:

    try:
        with engine.connect() as conn:
            logger.info("Connection established")
            create_table(engine, table_name)

            if push:
                push_data(conn)
                return 1
            return 0
    except OperationalError as e:
        logger.error(f"Connection failed | {e}")
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
    logger.info("Pushing data")
    pd.read_parquet(file_path).to_sql(
        table_name,
        con=conn,
        if_exists="append",
        index=False,
        method="multi",
    )

    logger.info("Data pushed to database")


def main():
    try:
        with create_db_engine() as engine:
            logger.info("Database engine is ready for use")

            with engine.connect() as conn:
                logger.info("Database connection established")
                query = "SELECT * FROM crowdedness ORDER BY timestamp DESC LIMIT 20;"

                df = pd.read_sql_query(query, conn)
                print(df)

    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")


if __name__ == "__main__":
    main()
