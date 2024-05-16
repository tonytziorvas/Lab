# Users/tonytziorvas/Documents/Code/Work/Lab/draft.py

import time
from io import TextIOWrapper
from typing import Dict, List, Union
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests

API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

REFRESH_INTERVAL = 60


# TODO : Make the script device-agnostic
def make_api_request(input_file: str, output_file: str) -> None:
    """
    Function to make an API request and save the data.

    This function makes a request to an external API and saves the
    retrieved data to a CSV file. The function runs in a loop with a
    REFRESH_INTERVAL, printing status messages at each iteration.

    Raises:
        requests.RequestException: If an exception occurs during the request.
    """
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
                print(
                    f"API request successful at time {time.ctime(timestamp)}, processing data..."
                )
                data = response.json()["vehicles_in_public_space"]
                tmp = parse_json_data(data, timestamp)
                df = pd.concat([df, tmp], ignore_index=True)
                print(f"Data processed, saved {len(tmp)} rows")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")

            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file} with {len(df)} total rows")

        print("Waiting for next refresh...\n")
        time.sleep(REFRESH_INTERVAL)


def parse_json_data(
    json_data: List[Dict[str, Union[str, float]]], timestamp: float
) -> pd.DataFrame:
    """
    Parse JSON data and convert it to a GeoDataFrame.

    Parameters:
    json_data (List[Dict[str, Union[str, float]]]): The JSON data to be parsed.

    Returns:
    gpd.GeoDataFrame: The parsed GeoDataFrame.
    """
    print("Parsing JSON data...")

    df = pd.DataFrame.from_dict(json_data)
    df["timestamp"] = timestamp
    df[["latitude", "longitude"]] = pd.DataFrame(df["location"].tolist())[
        ["latitude", "longitude"]
    ]

    df = df.drop("location", axis=1)
    print("Adding latitude and longitude columns and dropping location column")

    return df


def read_zip(zip_file_path: str, csv_name: str) -> pd.DataFrame:
    """
    Read a CSV file from a ZIP archive.

    Args:
        zip_file_path: The path to the ZIP archive.
        csv_name: The name of the CSV file within the archive.

    Returns:
        A Pandas DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the CSV file is not found in the archive.
    """
    with ZipFile(zip_file_path, "r") as zip_file:
        # Check if the CSV file exists in the zip archive
        if csv_name not in zip_file.namelist():
            raise FileNotFoundError(
                f"CSV file '{csv_name}' not found in the zip archive."
            )

        # Open the CSV file within the zip archive
        with zip_file.open(csv_name, "r") as csv_file:
            # Read the CSV file using an io.TextIOWrapper to handle decoding
            csv_text_wrapper = TextIOWrapper(csv_file, encoding="utf-8")
            # Create a pandas DataFrame from the CSV data
            df = pd.read_csv(csv_text_wrapper)

    return df


def get_points_in_boundary(
    city: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame, ts_col: str
) -> gpd.GeoDataFrame:
    """
    Find the points within the boundary of a given city.

    Parameters:
    city (gpd.GeoDataFrame): The city boundary.
    gdf (gpd.GeoDataFrame): The GeoDataFrame containing the points.

    Returns:
    gpd.GeoDataFrame: The points within the city boundary, with additional timestamp information.
    """

    # Query the spatial index of the GeoDataFrame to find the points within the boundary
    df_left = (
        pd.DataFrame(
            data=gdf.sindex.query(city.geometry, predicate="intersects").T,
            columns=["district_id", "point_id"],
        )
        .sort_values(by="point_id")
        .reset_index(drop=True)
    )

    # Get the timestamp information for each point
    df_right = (
        gdf.iloc[df_left["point_id"]][ts_col]
        .reset_index()
        .rename(columns={"index": "point_id", ts_col: "timestamp"})
    )

    # Merge the points with their timestamp information
    points = pd.merge(df_left, df_right, on="point_id")

    return points


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
    make_api_request(args.input_file, args.output_file)
