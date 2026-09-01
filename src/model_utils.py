
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Geod
import shap

from sklearn.base import BaseEstimator, TransformerMixin

from src.geo_utils import (
    prepare_geospatial_resources,
    build_geospatial_features,
)

GEOD = Geod(ellps="WGS84")


# ============================================================
# TRANSFORMER NECESARIO PARA CARGAR EL PIPELINE
# ============================================================

class H3TargetMeanEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        h3_columns=("h3_res8", "h3_res9"),
        smoothing=11.720095,
        drop_original=True,
    ):
        self.h3_columns = h3_columns
        self.smoothing = smoothing
        self.drop_original = drop_original

    def fit(self, X, y):

        X = (
            pd.DataFrame(X)
            .reset_index(drop=True)
            .copy()
        )

        y = (
            pd.Series(y)
            .reset_index(drop=True)
        )

        self.global_mean_ = float(
            y.mean()
        )

        self.encoding_maps_ = {}

        for col in self.h3_columns:

            if col not in X.columns:
                continue

            tmp = pd.DataFrame({
                "group": (
                    X[col]
                    .astype("string")
                    .fillna("H3_desconocido")
                ),
                "target": y,
            })

            stats = (
                tmp.groupby("group")["target"]
                .agg(["mean", "count"])
            )

            smooth = (
                stats["count"] * stats["mean"]
                + self.smoothing
                * self.global_mean_
            ) / (
                stats["count"]
                + self.smoothing
            )

            self.encoding_maps_[col] = (
                smooth.to_dict()
            )

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        for col in self.h3_columns:

            if col not in X.columns:
                continue

            X[
                f"{col}_mean_price_te"
            ] = (
                X[col]
                .astype("string")
                .fillna("H3_desconocido")
                .map(
                    self.encoding_maps_
                    .get(col, {})
                )
                .fillna(
                    self.global_mean_
                )
                .astype(float)
            )

        if self.drop_original:

            X = X.drop(
                columns=[
                    col
                    for col
                    in self.h3_columns
                    if col in X.columns
                ],
                errors="ignore",
            )

        return X


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_model_final.pkl"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "listings_features.parquet"
)

DISTRICTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "neighbourhoods.geojson"
)


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================

df_features = pd.read_parquet(
    FEATURES_PATH
)

X_app = (
    df_features
    .drop(
        columns=["price"],
        errors="ignore",
    )
    .copy()
)


# Importante:
# el pickle original referencia H3TargetMeanEncoder
# como clase definida en __main__.
#
# Para mantener compatibilidad con el modelo ya guardado,
# exponemos temporalmente la clase en __main__ antes de cargar.

import __main__

__main__.H3TargetMeanEncoder = (
    H3TargetMeanEncoder
)


model = joblib.load(
    MODEL_PATH
)


# ============================================================
# RECURSOS GEOESPACIALES
# ============================================================

geo_resources = (
    prepare_geospatial_resources(
        districts_file=DISTRICTS_FILE,
        df_features=df_features,
    )
)


# ============================================================
# VARIABLES DE LA INTERFAZ
# ============================================================

USER_AMENITIES = [
    "has_wifi",
    "has_kitchen",
    "has_air_conditioning",
    "has_heating",
    "has_parking",
    "has_pool",
    "has_washer",
    "has_dryer",
    "has_tv",
    "has_balcony_or_terrace",
    "has_sea_view",
    "has_workspace",
    "has_elevator",
    "has_pets_allowed",
    "has_crib",
    "has_bbq",
    "has_gym",
    "has_hot_tub",
    "has_breakfast",
]


HISTORICAL_COLUMNS = [
    "host_listings_count",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "availability_30",
    "availability_365",
]


historical_defaults = {
    col: X_app[col].median()
    for col in HISTORICAL_COLUMNS
}


# ============================================================
# REFERENCIAS INTERNAS DEL PIPELINE
# ============================================================

h3_encoder = (
    model.named_steps[
        "h3_target_encoder"
    ]
)

preprocessor = (
    model.named_steps[
        "preprocessor"
    ]
)

variance_filter = (
    model.named_steps[
        "variance_filter"
    ]
)

xgboost_model = (
    model.named_steps[
        "model"
    ]
)


feature_names_pre = (
    preprocessor
    .get_feature_names_out()
)

variance_mask = (
    variance_filter
    .get_support()
)

feature_names_final = (
    feature_names_pre[
        variance_mask
    ]
)


explainer = shap.TreeExplainer(
    xgboost_model
)


# ============================================================
# CONSTRUCCIÓN DE LAS 164 VARIABLES
# ============================================================

def build_listing_features(
    latitude,
    longitude,
    room_type,
    property_type_group,
    accommodates,
    bedrooms,
    bathrooms_num,
    minimum_nights,
    maximum_nights,
    instant_bookable,
    host_is_superhost,
    selected_amenities=None,
):

    if selected_amenities is None:
        selected_amenities = []

    new_row = {}

    # --------------------------------------------------------
    # Valores neutrales iniciales
    # --------------------------------------------------------

    numeric_columns = (
        X_app
        .select_dtypes(
            include=["number"]
        )
        .columns
    )

    for col in numeric_columns:
        new_row[col] = (
            X_app[col].median()
        )

    categorical_columns = (
        X_app
        .select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
    )

    for col in categorical_columns:

        mode = (
            X_app[col]
            .mode(dropna=True)
        )

        new_row[col] = (
            mode.iloc[0]
            if len(mode) > 0
            else np.nan
        )

    # --------------------------------------------------------
    # Datos del usuario
    # --------------------------------------------------------

    new_row.update({
        "latitude":
            float(latitude),

        "longitude":
            float(longitude),

        "room_type":
            room_type,

        "property_type_group":
            property_type_group,

        "accommodates":
            int(accommodates),

        "bedrooms":
            float(bedrooms),

        "bathrooms_num":
            float(bathrooms_num),

        "minimum_nights":
            int(minimum_nights),

        "maximum_nights":
            int(maximum_nights),

        "instant_bookable":
            int(instant_bookable),

        "host_is_superhost":
            int(host_is_superhost),
    })

    # --------------------------------------------------------
    # Variables históricas
    # --------------------------------------------------------

    for col, value in (
        historical_defaults.items()
    ):
        new_row[col] = value

    # --------------------------------------------------------
    # Variables geoespaciales
    # --------------------------------------------------------

    geo_features = (
        build_geospatial_features(
            latitude=latitude,
            longitude=longitude,
            resources=geo_resources,
            new_listing=True,
        )
    )

    new_row.update(
        geo_features
    )

    # --------------------------------------------------------
    # Amenities manuales
    # --------------------------------------------------------

    for col in USER_AMENITIES:

        new_row[col] = int(
            col in selected_amenities
        )

    new_row[
        "amenities_count"
    ] = len(
        selected_amenities
    )

    # --------------------------------------------------------
    # Amenities automáticas
    # --------------------------------------------------------

    amenity_auto_columns = [
        col
        for col in X_app.columns
        if col.startswith(
            "amenity_auto_"
        )
    ]

    for col in amenity_auto_columns:
        new_row[col] = 0


    amenity_auto_map = {

        "has_wifi":
            "amenity_auto_wifi",

        "has_kitchen":
            "amenity_auto_kitchen",

        "has_air_conditioning":
            "amenity_auto_air_conditioning",

        "has_heating":
            "amenity_auto_heating",

        "has_washer":
            "amenity_auto_washer",

        "has_tv":
            "amenity_auto_tv",

        "has_elevator":
            "amenity_auto_elevator",

        "has_workspace":
            "amenity_auto_dedicated_workspace",

        "has_crib":
            "amenity_auto_crib",
    }


    for (
        manual_col,
        auto_col,
    ) in amenity_auto_map.items():

        if (
            manual_col
            in selected_amenities
            and auto_col
            in new_row
        ):
            new_row[
                auto_col
            ] = 1


    new_row[
        "amenities_count_phase2"
    ] = sum(
        new_row[col]
        for col
        in amenity_auto_columns
    )


    # --------------------------------------------------------
    # Variables derivadas
    # --------------------------------------------------------

    new_row[
        "has_reviews"
    ] = int(
        new_row[
            "number_of_reviews"
        ] > 0
    )


    # --------------------------------------------------------
    # Orden exacto esperado por el modelo
    # --------------------------------------------------------

    X_new = pd.DataFrame(
        [new_row]
    )

    X_new = (
        X_new
        .reindex(
            columns=X_app.columns
        )
    )

    return X_new


# ============================================================
# PREDICCIÓN FINAL
# ============================================================

def predict_listing(
    latitude,
    longitude,
    room_type,
    property_type_group,
    accommodates,
    bedrooms,
    bathrooms_num,
    minimum_nights,
    maximum_nights,
    instant_bookable,
    host_is_superhost,
    selected_amenities=None,
):

    X_new = build_listing_features(
        latitude=latitude,
        longitude=longitude,
        room_type=room_type,
        property_type_group=property_type_group,
        accommodates=accommodates,
        bedrooms=bedrooms,
        bathrooms_num=bathrooms_num,
        minimum_nights=minimum_nights,
        maximum_nights=maximum_nights,
        instant_bookable=instant_bookable,
        host_is_superhost=host_is_superhost,
        selected_amenities=selected_amenities,
    )

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    pred_price = float(
        model.predict(
            X_new
        )[0]
    )

    # --------------------------------------------------------
    # Transformación para SHAP
    # --------------------------------------------------------

    X_h3 = (
        h3_encoder
        .transform(
            X_new
        )
    )

    X_pre = (
        preprocessor
        .transform(
            X_h3
        )
    )

    X_var = (
        variance_filter
        .transform(
            X_pre
        )
    )

    X_var_dense = (
        X_var.toarray()
        if hasattr(
            X_var,
            "toarray",
        )
        else np.asarray(
            X_var
        )
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = (
        explainer(
            X_var_dense
        )
    )

    shap_explanation = (
        shap.Explanation(
            values=(
                shap_values
                .values[0]
            ),
            base_values=(
                shap_values
                .base_values[0]
            ),
            data=(
                X_var_dense[0]
            ),
            feature_names=(
                feature_names_final
            ),
        )
    )

    return {
        "price":
            pred_price,

        "features":
            X_new,

        "shap_explanation":
            shap_explanation,
    }

# ============================================================
# DATOS PARA EL VISOR CARTOGRÁFICO
# ============================================================

def get_map_context(
    latitude,
    longitude,
    radius_km=1.0,
):
    """
    Devuelve la información necesaria para representar la vivienda,
    su distrito, su celda H3 res. 8, los competidores de dicha celda
    y los alojamientos situados dentro de un radio determinado.
    """

    latitude = float(latitude)
    longitude = float(longitude)
    radius_km = float(radius_km)

    geo_features = build_geospatial_features(
        latitude=latitude,
        longitude=longitude,
        resources=geo_resources,
        new_listing=True,
    )

    district = geo_features["district"]
    h3_res8 = geo_features["h3_res8"]

    district_gdf = (
        geo_resources["gdf_districts"]
        .loc[
            geo_resources["gdf_districts"]["district"] == district
        ]
        .copy()
    )

    competitors = (
        df_features
        .loc[
            df_features["h3_res8"] == h3_res8,
            [
                "latitude",
                "longitude",
                "price",
                "room_type",
                "property_type_group",
            ],
        ]
        .copy()
    )

    nearby = df_features[
        [
            "latitude",
            "longitude",
            "price",
            "room_type",
            "property_type_group",
        ]
    ].copy()

    _, _, distances_m = GEOD.inv(
        np.full(len(nearby), longitude),
        np.full(len(nearby), latitude),
        nearby["longitude"].to_numpy(),
        nearby["latitude"].to_numpy(),
    )

    nearby["distance_km"] = distances_m / 1000.0

    nearby = (
        nearby
        .loc[nearby["distance_km"] <= radius_km]
        .sort_values("distance_km")
        .copy()
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "district": district,
        "h3_res8": h3_res8,
        "district_gdf": district_gdf,
        "competitors": competitors,
        "nearby": nearby,
        "radius_km": radius_km,
    }
