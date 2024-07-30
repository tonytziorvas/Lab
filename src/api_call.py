import time
from io import TextIOWrapper
from typing import Dict, List, Union
from zipfile import ZipFile

import pandas as pd
import requests

from utils import logger

API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

REFRESH_INTERVAL = 60


# def _parse_json_data(
#     json_data: List[Dict[str, Union[str, float]]], timestamp: float
# ) -> pd.DataFrame:
#     """
#     Parse JSON data and convert it to a GeoDataFrame.

#     Parameters:
#     json_data (List[Dict[str, Union[str, float]]]): The JSON data to be parsed.

#     Returns:
#     gpd.GeoDataFrame: The parsed GeoDataFrame.
#     """
#     print("\nParsing JSON data")

#     df = pd.DataFrame(json_data)
#     df["timestamp"] = timestamp
#     df[["latitude", "longitude"]] = pd.DataFrame(df["location"].tolist())[
#         ["latitude", "longitude"]
#     ]
#     df = df.drop(columns="location", axis=1)
#     print("Added lat/lon columns and dropping location column\n")


#     return df
def _parse_json_data(
    json_data: List[Dict[str, Union[str, float]]], timestamp: float
) -> pd.DataFrame:

    print("\nParsing JSON data")

    df = pd.DataFrame(json_data)
    df["timestamp"] = timestamp
    df["latitude"] = df["location"].apply(lambda loc: loc["latitude"])
    df["longitude"] = df["location"].apply(lambda loc: loc["longitude"])
    df = df.drop(columns="location", axis=1)
    print("Added lat/lon columns and dropping location column\n")

    return df


# TODO : Make the script device-agnostic
def fetch_data(input_file: str | None, output_file: str) -> None:
    """The function `fetch_data` retrieves data from an API at regular intervals
    processes the data, and saves it to an output file.

    Parameters
    ----------
    input_file : str
        The `input_file` parameter in the `fetch_data` function is a string
        that represents the file path to a CSV file containing data that will be used
        as input for the API request
    output_file : str
        The `output_file` parameter in the `fetch_data` function is a string
        that represents the file path where the processed data will be saved to as
        a CSV file. This file will contain the data retrieved from the API and
        processed during the execution of the function.

    """

    df = pd.DataFrame() if input_file is None else pd.read_csv(input_file)
    logging.info("Starting data retrieval process")
    logging.info(f"Refresh rate is {REFRESH_INTERVAL} seconds")

    while True:
        with requests.get(API_URL) as response:

            if response.status_code == 200:
                ts = time.time()
                logging.info(
                    f"Request successful | status code: {response.status_code}"
                )
                data = response.json()["vehicles_in_public_space"]
                tmp = _parse_json_data(data, ts)
                df = pd.concat([df, tmp], ignore_index=True)
                logging.info(f"{len(tmp)} rows processed")
            else:
                logging.error(
                    f"Failed to retrieve data. Status code: {response.status_code}"
                )

            df.to_csv(output_file, index=False)
            logging.info(f"Data saved to {output_file} with {len(df)} total rows")

        logging.info("Waiting for next refresh\n")
        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    logging = logger.setup_logger("api_call")
    import argparse

    parser = argparse.ArgumentParser(
        description="Make API request and save data to CSV."
    )
    parser.add_argument(
        "--input_file",
        help="Specify an input file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--output_file",
        help="Specify an output file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    fetch_data(args.input_file, args.output_file)
