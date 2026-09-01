
import re
import unicodedata

import h3
import numpy as np
import pandas as pd
import geopandas as gpd

from pyproj import Geod
from shapely.geometry import Point


# ============================================================
# POIs UTILIZADOS EN EL ENTRENAMIENTO
# ============================================================

POIS = {
    "malagueta": (
        36.7196,
        -4.4087,
        "beach",
    ),
    "pedregalejo": (
        36.7209,
        -4.3699,
        "beach",
    ),
    "el_palo": (
        36.7169,
        -4.3565,
        "beach",
    ),
    "calle_larios": (
        36.7196,
        -4.4211,
        "historic_center",
    ),
    "plaza_constitucion": (
        36.7202,
        -4.4218,
        "historic_center",
    ),
    "museo_picasso": (
        36.7217,
        -4.4185,
        "culture",
    ),
    "alcazaba": (
        36.7213,
        -4.4165,
        "culture",
    ),
    "centro_pompidou": (
        36.7182,
        -4.4130,
        "culture",
    ),
    "maria_zambrano": (
        36.7111,
        -4.4314,
        "transport",
    ),
    "aeropuerto_agp": (
        36.6749,
        -4.4991,
        "transport",
    ),
}


GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# UTILIDADES
# ============================================================

def normalize_column_name(column_name):

    text = str(column_name).strip().lower()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def mode_or_nan(series):

    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return series.mode().iloc[0]


# ============================================================
# PREPARACIÓN DE RECURSOS
# ============================================================

def prepare_geospatial_resources(
    districts_file,
    df_features,
):

    gdf_raw = gpd.read_file(
        districts_file
    )

    normalized_column_map = {
        normalize_column_name(column): column
        for column in gdf_raw.columns
    }

    district_name_candidates = [
        "neighbourhood",
        "neighbourhood_group",
        "nombre",
        "distrito",
        "nom_distrito",
        "nombre_distrito",
        "descrip",
        "descripcion",
        "name",
    ]

    district_name_column = None

    for candidate in district_name_candidates:

        if candidate in normalized_column_map:

            district_name_column = (
                normalized_column_map[candidate]
            )

            break

    if district_name_column is None:
        raise KeyError(
            "No se ha podido detectar la "
            "columna de distrito."
        )

    gdf_districts = (
        gdf_raw[
            [
                district_name_column,
                gdf_raw.geometry.name,
            ]
        ]
        .rename(
            columns={
                district_name_column:
                "district"
            }
        )
        .copy()
    )

    gdf_districts["district"] = (
        gdf_districts["district"]
        .astype(str)
        .str.strip()
    )

    if gdf_districts.crs is None:

        gdf_districts = (
            gdf_districts
            .set_crs("EPSG:4326")
        )

    gdf_districts = (
        gdf_districts
        .to_crs("EPSG:4326")
    )

    # Elimina dimensión Z si existe
    try:
        import shapely

        gdf_districts["geometry"] = (
            gdf_districts.geometry.apply(
                lambda geometry:
                shapely.force_2d(geometry)
                if getattr(
                    geometry,
                    "has_z",
                    False,
                )
                else geometry
            )
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # ÁREA DE DISTRITOS
    # --------------------------------------------------------

    metric = (
        gdf_districts
        .to_crs("EPSG:25830")
        .copy()
    )

    metric[
        "district_area_km2"
    ] = (
        metric.geometry.area
        / 1_000_000
    )

    district_area_map = (
        metric
        .set_index("district")[
            "district_area_km2"
        ]
        .to_dict()
    )

    # --------------------------------------------------------
    # FRECUENCIAS DEL SNAPSHOT
    # --------------------------------------------------------

    district_count_map = (
        df_features
        .groupby("district")
        .size()
        .to_dict()
    )

    neighbourhood_map = (
        df_features
        .groupby("district")[
            "neighbourhood_cleansed"
        ]
        .agg(mode_or_nan)
        .to_dict()
    )

    neighbourhood_group_map = (
        df_features
        .groupby("district")[
            "neighbourhood_group"
        ]
        .agg(mode_or_nan)
        .to_dict()
    )

    h3_count_maps = {}

    for resolution in [8, 9]:

        column = (
            f"h3_res{resolution}"
        )

        h3_count_maps[
            resolution
        ] = (
            df_features[column]
            .value_counts()
            .to_dict()
        )

    return {
        "gdf_districts": gdf_districts,
        "district_area_map":
            district_area_map,
        "district_count_map":
            district_count_map,
        "neighbourhood_map":
            neighbourhood_map,
        "neighbourhood_group_map":
            neighbourhood_group_map,
        "h3_count_maps":
            h3_count_maps,
    }


# ============================================================
# CONSTRUCCIÓN DE FEATURES GEOESPACIALES
# ============================================================

def build_geospatial_features(
    latitude,
    longitude,
    resources,
    new_listing=True,
):

    features = {
        "latitude": float(latitude),
        "longitude": float(longitude),
    }

    gdf_districts = (
        resources["gdf_districts"]
    )

    # --------------------------------------------------------
    # DISTRITO
    # --------------------------------------------------------

    point_gdf = gpd.GeoDataFrame(
        {
            "latitude": [latitude],
            "longitude": [longitude],
        },
        geometry=[
            Point(
                float(longitude),
                float(latitude),
            )
        ],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        point_gdf,
        gdf_districts[
            [
                "district",
                "geometry",
            ]
        ],
        how="left",
        predicate="within",
    )

    district = (
        joined.iloc[0]["district"]
    )

    if pd.isna(district):

        raise ValueError(
            "La ubicación no pertenece a "
            "ningún distrito disponible."
        )

    features["district"] = district

    features[
        "neighbourhood_cleansed"
    ] = (
        resources[
            "neighbourhood_map"
        ].get(
            district,
            district,
        )
    )

    features[
        "neighbourhood_group"
    ] = (
        resources[
            "neighbourhood_group_map"
        ].get(
            district,
            district,
        )
    )

    # --------------------------------------------------------
    # DENSIDAD DISTRITO
    # --------------------------------------------------------

    existing_count = int(
        resources[
            "district_count_map"
        ].get(
            district,
            0,
        )
    )

    listing_count = (
        existing_count + 1
        if new_listing
        else existing_count
    )

    district_area = float(
        resources[
            "district_area_map"
        ][district]
    )

    features[
        "district_listing_count"
    ] = listing_count

    features[
        "district_area_km2"
    ] = district_area

    features[
        "district_listing_density_km2"
    ] = (
        listing_count
        / district_area
    )

    # --------------------------------------------------------
    # H3
    # --------------------------------------------------------

    for resolution in [8, 9]:

        cell = h3.latlng_to_cell(
            float(latitude),
            float(longitude),
            resolution,
        )

        features[
            f"h3_res{resolution}"
        ] = cell

        existing_h3_count = int(
            resources[
                "h3_count_maps"
            ][resolution].get(
                cell,
                0,
            )
        )

        if new_listing:

            h3_listing_count = (
                existing_h3_count + 1
            )

            competitor_count = (
                existing_h3_count
            )

        else:

            h3_listing_count = (
                existing_h3_count
            )

            competitor_count = max(
                h3_listing_count - 1,
                0,
            )

        area = h3.cell_area(
            cell,
            unit="km^2",
        )

        features[
            f"h3_res{resolution}_listing_count"
        ] = h3_listing_count

        features[
            f"h3_res{resolution}_competitor_count"
        ] = competitor_count

        features[
            f"h3_res{resolution}_area_km2"
        ] = area

        features[
            f"h3_res{resolution}_density_km2"
        ] = (
            h3_listing_count
            / area
        )

    # --------------------------------------------------------
    # DISTANCIAS POI
    # --------------------------------------------------------

    for poi_name, (
        poi_latitude,
        poi_longitude,
        poi_category,
    ) in POIS.items():

        _, _, distance_m = GEOD.inv(
            float(longitude),
            float(latitude),
            float(poi_longitude),
            float(poi_latitude),
        )

        features[
            f"distance_{poi_name}_km"
        ] = (
            distance_m / 1000
        )

    # --------------------------------------------------------
    # POI MÁS CERCANO
    # --------------------------------------------------------

    categories = sorted({
        values[2]
        for values in POIS.values()
    })

    for category in categories:

        category_pois = [
            name
            for name, values
            in POIS.items()
            if values[2] == category
        ]

        distances = {
            name: features[
                f"distance_{name}_km"
            ]
            for name in category_pois
        }

        nearest_poi = min(
            distances,
            key=distances.get,
        )

        features[
            f"distance_nearest_{category}_km"
        ] = distances[
            nearest_poi
        ]

        features[
            f"nearest_{category}_poi"
        ] = nearest_poi

    # --------------------------------------------------------
    # INDICADORES DE PROXIMIDAD
    # --------------------------------------------------------

    distance_thresholds = {
        "within_500m_beach": (
            "distance_nearest_beach_km",
            0.5,
        ),
        "within_1km_beach": (
            "distance_nearest_beach_km",
            1.0,
        ),
        "within_1km_historic_center": (
            "distance_nearest_historic_center_km",
            1.0,
        ),
        "within_2km_historic_center": (
            "distance_nearest_historic_center_km",
            2.0,
        ),
        "within_1km_culture": (
            "distance_nearest_culture_km",
            1.0,
        ),
        "within_1km_transport": (
            "distance_nearest_transport_km",
            1.0,
        ),
        "within_5km_airport": (
            "distance_aeropuerto_agp_km",
            5.0,
        ),
    }

    for new_column, (
        distance_column,
        threshold,
    ) in distance_thresholds.items():

        features[new_column] = int(
            features[
                distance_column
            ]
            <= threshold
        )

    # --------------------------------------------------------
    # LOG1P
    # --------------------------------------------------------

    distance_columns = [
        column
        for column in list(
            features.keys()
        )
        if (
            column.startswith(
                "distance_"
            )
            and column.endswith(
                "_km"
            )
        )
    ]

    for column in distance_columns:

        features[
            f"log1p_{column}"
        ] = np.log1p(
            features[column]
        )

    return features
