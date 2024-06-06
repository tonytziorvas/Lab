# Users/tonytziorvas/Documents/Code/Work/Lab/draft.py

import time
from io import TextIOWrapper
from typing import Dict, List, Union
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate

# from sklearn.preprocessing import FunctionTransformer

API_URL = (
    "https://api.deelfietsdashboard.nl/dashboard-api/public/vehicles_in_public_space"
)

REFRESH_INTERVAL = 60


# TODO : Make the script device-agnostic
def make_api_request(input_file: str, output_file: str) -> None:
    """The function `make_api_request` retrieves data from an API at regular intervals
    processes the data, and saves it to an output file.

    Parameters
    ----------
    input_file : str
        The `input_file` parameter in the `make_api_request` function is a string
        that represents the file path to a CSV file containing data that will be used
        as input for the API request
    output_file : str
        The `output_file` parameter in the `make_api_request` function is a string
        that represents the file path where the processed data will be saved to as
        a CSV file. This file will contain the data retrieved from the API and
        processed during the execution of the function.

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


def get_points_in_boundary(city, gdf, ts_col):
    """The function `get_points_in_boundary` retrieves points within a city boundary
    and maps district IDs to district names.

    Parameters
    ----------
    city
        A string representing the name of the city for which
        you want to get points within the boundary.
    gdf
        `gdf` is a GeoDataFrame containing points data with spatial information such as
        geometry coordinates.
    ts_col
        The column name in the GeoDataFrame (`gdf`) that contains the
        timestamps associated with each point. This column is used to retrieve the
        timestamps for the points that intersect with the city boundaries.

    Returns
    -------
        The function `get_points_in_boundary` returns two objects:
        `points` and `city_boundaries`.

    """
    city_boundaries = gpd.read_file(f"data/boundaries/{city}_.geojson")

    # Query points that intersect with city boundaries

    df_left = (
        pd.DataFrame(
            data=gdf.sindex.query(city_boundaries.geometry, predicate="intersects").T,
            columns=["district_id", "point_id"],
        )
        .sort_values(by="point_id")
        .reset_index(drop=True)
    )

    # Get corresponding timestamp and district_id for points
    df_right = (
        gdf.iloc[df_left["point_id"]][ts_col]
        .reset_index()
        .rename(columns={"index": "point_id", ts_col: "timestamp"})
    )

    # Merge the two dataframes based on point_id
    points = pd.merge(df_left, df_right, on="point_id")

    # Map district_id to district names
    district_codes = dict(city_boundaries.iloc[points.district_id.unique()]["name"])
    points["district_id"] = points["district_id"].map(district_codes)

    return points, city_boundaries


# TODO Replace with FunctionTransformer of sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
def cyclic_encode(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    Encode a column of a dataframe with sine and cosine functions
    based on the period.

    Parameters
    ----------
    df (pd.DataFrame): The dataframe to encode.
    col (str): The name of the column to encode.
    period (int): The period of the sine and cosine functions.

    Returns:
        pd.DataFrame: The encoded dataframe.
    """
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df


def label_smoothing(labels, epsilon=0.1):
    """The function `label_smoothing` applies label smoothing to input labels with a
    specified epsilon value.

    Parameters
    ----------
    labels
        The `labels` parameter is expected to be a 2D array representing one-hot
        encoded labels for classification tasks. Each row in the array corresponds to a
        sample, and each column represents a class with a binary indicator
        (1 for the correct class, 0 for others).
    epsilon
        The `epsilon` parameter in the `label_smoothing` function represents the
        smoothing factor that is used to adjust the labels. It is a hyperparameter
        that controls the amount of smoothing applied to the labels. A higher value of
        `epsilon` results in more smoothing, while a lower value preserves the original

    Returns
    -------
        The function `label_smoothing` returns the labels after applying label
        smoothing with a given epsilon value.

    """
    num_classes = labels.shape[1]
    return (1 - epsilon) * labels + epsilon / num_classes


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
