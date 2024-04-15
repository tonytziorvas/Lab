# Users/tonytziorvas/Documents/Code/Work/Lab/draft.py
import os
import time
from typing import Dict, List, Union

import geopandas as gpd
import pandas as pd
import requests

API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

OUTPUT_INDEX = len(os.listdir(path="data/raw")) + 1
DEFAULT_OUTPUT_FILE = f"data/raw/data_{OUTPUT_INDEX}.csv"
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
        try:
            print("Making API request...")
            response = requests.get(API_URL)
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
        except requests.RequestException as e:
            print(f"Error occurred: {str(e)}")

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

    parser = argparse.ArgumentParser()
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
        required=False,
        default=DEFAULT_OUTPUT_FILE,
    )
    args = parser.parse_args()
    print(f"Input File: {args.input_file} --- Destination File: {args.output_file}")
    make_api_request(args.input_file, args.output_file)
