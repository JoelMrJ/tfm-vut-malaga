from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.model_utils import predict_listing
from src.rag_utils import ask_rag


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="VUT Málaga API",
    description=(
        "API REST para el motor predictivo XGBoost y el "
        "consultor documental RAG del TFM."
    ),
    version="1.1.0",
)


# ============================================================
# ESQUEMAS DE ENTRADA
# ============================================================

class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    room_type: str
    property_type_group: str

    accommodates: int = Field(..., ge=1, le=16)
    bedrooms: int = Field(..., ge=0, le=15)
    bathrooms_num: float = Field(..., ge=0, le=10)

    minimum_nights: int = Field(..., ge=1)
    maximum_nights: int = Field(..., ge=1)

    instant_bookable: int = Field(..., ge=0, le=1)
    host_is_superhost: int = Field(..., ge=0, le=1)

    selected_amenities: list[str] = []


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=3)
    n_results: int = Field(default=4, ge=1, le=10)


# ============================================================
# UTILIDADES DE SERIALIZACIÓN
# ============================================================

def _to_python(value: Any):
    """Convierte tipos NumPy/Pandas a tipos JSON estándar."""

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def _row_to_dict(row: pd.Series) -> dict:
    return {
        str(key): _to_python(value)
        for key, value in row.items()
    }


def _serialize_shap(explanation) -> dict:
    values = np.asarray(
        explanation.values,
        dtype=float,
    ).reshape(-1)

    base_value = np.asarray(
        explanation.base_values,
        dtype=float,
    ).reshape(-1)

    feature_names = [
        str(name)
        for name in explanation.feature_names
    ]

    data = np.asarray(
        explanation.data,
        dtype=object,
    ).reshape(-1)

    return {
        "base_value": float(base_value[0]),
        "values": [
            float(value)
            for value in values
        ],
        "feature_names": feature_names,
        "data": [
            _to_python(value)
            for value in data
        ],
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VUT Málaga API",
        "version": "1.1.0",
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "target": "price",
        "objective": "reg:absoluteerror",
        "validation": "holdout espacial H3 res8 80/20",
        "metrics": {
            "r2": 0.529734,
            "rmse_eur": 57.941030,
            "mae_eur": 33.202631,
            "mape_pct": 24.054230,
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        result = predict_listing(
            latitude=request.latitude,
            longitude=request.longitude,
            room_type=request.room_type,
            property_type_group=request.property_type_group,
            accommodates=request.accommodates,
            bedrooms=request.bedrooms,
            bathrooms_num=request.bathrooms_num,
            minimum_nights=request.minimum_nights,
            maximum_nights=request.maximum_nights,
            instant_bookable=request.instant_bookable,
            host_is_superhost=request.host_is_superhost,
            selected_amenities=request.selected_amenities,
        )

        feature_row = result["features"].iloc[0]

        response = {
            "price": float(result["price"]),
            "features": _row_to_dict(feature_row),
            "shap": _serialize_shap(
                result["shap_explanation"]
            ),
        }

        return jsonable_encoder(response)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {exc}",
        ) from exc


@app.post("/rag")
def rag(request: RAGRequest):

    try:
        result = ask_rag(
            request.question,
            n_results=request.n_results,
        )

        sources = result["sources"]

        if sources is None:
            source_records = []

        elif isinstance(sources, pd.DataFrame):
            source_records = [
                {
                    str(key): _to_python(value)
                    for key, value in row.items()
                }
                for row in sources.to_dict(
                    orient="records"
                )
            ]

        else:
            source_records = sources

        response = {
            "question": request.question,
            "answer": result["answer"],
            "sources": source_records,
        }

        return jsonable_encoder(response)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el sistema RAG: {exc}",
        ) from exc
