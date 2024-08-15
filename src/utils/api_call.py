import logging
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import requests

REFRESH_INTERVAL = 60
API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

logger = logging.getLogger("etl_pipeline")


def _parse_json_data(
    json_data: List[Dict[str, Union[str, float]]],
    timestamp: float,
) -> pd.DataFrame:

    logger.info("\nParsing JSON data")

    df = pd.DataFrame(json_data)
    df["ts"] = timestamp
    df["latitude"] = df["location"].apply(lambda row: row["latitude"])
    df["longitude"] = df["location"].apply(lambda row: row["longitude"])
    df = df.drop(columns="location", axis=1)
    logger.info("Added lat/lon columns and dropping location column\n")

    return df


def fetch_data() -> Optional[pd.DataFrame]:
    response = requests.get(API_URL)
    status_code = response.status_code

    if status_code == 200:
        ts = time.time()
        logger.info(f"Request successful | Status Code: {status_code}")

        if data := response.json().get("vehicles_in_public_space", []):
            data = _parse_json_data(data, ts)

            logger.info(f"{len(data)} rows processed")
            logger.info("Waiting for next refresh\n")
            return data
    else:
        logger.error(f"Request failed | Status code: {status_code}")
        return
