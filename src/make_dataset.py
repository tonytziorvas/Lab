import logging

import geopandas as gpd

from utils import helper

logging.basicConfig(
    format="[%(asctime)s] | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    # handlers=[logging.FileHandler("reports/make_dataset.log")],
)

city = "rotterdam"
OUTPUT_FILE = "data/final/points_per_district_full"
CHUNK_SIZE = 10**7
DTYPES = {
    "form_factor": "object",
    "system_id": "object",
    "longitude": "float64",
    "latitude": "float64",
    "ts": "int64",
}

city_boundaries = gpd.read_file(f"misc/{city}_.geojson")


if __name__ == "__main__":
    helper.etl_pipeline(OUTPUT_FILE, CHUNK_SIZE, DTYPES, city_boundaries)
