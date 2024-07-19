import time
from io import TextIOWrapper
from typing import Dict, List, Union
from zipfile import ZipFile

import pandas as pd
import requests

API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

REFRESH_INTERVAL = 60


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

    def _parse_json_data(
        json_data: List[Dict[str, Union[str, float]]], timestamp: float
    ) -> pd.DataFrame:
        """
        Parse JSON data and convert it to a GeoDataFrame.

        Parameters:
        json_data (List[Dict[str, Union[str, float]]]): The JSON data to be parsed.

        Returns:
        gpd.GeoDataFrame: The parsed GeoDataFrame.
        """
        print("Parsing JSON data")

        df = pd.DataFrame(json_data)
        df["timestamp"] = timestamp
        df[["latitude", "longitude"]] = pd.DataFrame(df["location"].tolist())[
            ["latitude", "longitude"]
        ]
        df.drop("location", axis=1, inplace=True)
        print("Added lat/lon columns and dropping location column")

        return df

    if input_file is None:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(input_file)

    start_time = time.time()
    print(f"Starting data retrieval process at {time.ctime(start_time)}")
    print(f"Refresh rate is {REFRESH_INTERVAL} seconds")

    while True:
        with requests.get(API_URL) as response:
            print("Making API request...")
            print(f"API request status code: {response.status_code}")

            if response.status_code == 200:
                timestamp = time.time()
                print(f"API request successful at time {time.ctime(timestamp)}\n")
                data = response.json()["vehicles_in_public_space"]
                tmp = _parse_json_data(data, timestamp)
                df = pd.concat([df, tmp], ignore_index=True)
                print(f"Data processed, saved {len(tmp)} rows")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")

            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file} with {len(df)} total rows")

        print("Waiting for next refresh\n")
        time.sleep(REFRESH_INTERVAL)


def read_zip(zip_file_path: str, csv_name: str) -> pd.DataFrame:
    """The function `read_zip` reads a CSV file from a zip archive and returns
    its contents as a pandas DataFrame.

    Parameters
    ----------
    zip_file_path: The file path to the ZIP archive from which
        you want to read the CSV file.
    csv_name: The name of the CSV file that you want to extract
    and read from the specified ZIP archive.

    Returns
    -------
        a DataFrame containing the data from the specified CSV file

    """
    with ZipFile(zip_file_path, "r") as zip_file:
        if csv_name not in zip_file.namelist():
            raise FileNotFoundError(
                f"CSV file '{csv_name}' not found in the zip archive."
            )

        with zip_file.open(csv_name, "r") as csv_file:
            # Read the CSV file using an io.TextIOWrapper to handle decoding
            csv_text_wrapper = TextIOWrapper(csv_file, encoding="utf-8")

            df = pd.read_csv(csv_text_wrapper)

    return df


if __name__ == "__main__":
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
    print(f"Input File: {args.input_file} --- Destination File: {args.output_file}")
    fetch_data(args.input_file, args.output_file)
