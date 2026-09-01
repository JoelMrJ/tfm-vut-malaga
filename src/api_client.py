import os

import requests


API_BASE_URL = os.getenv(
    "VUT_API_URL",
    "http://127.0.0.1:8000",
)


def health_api(timeout=10):
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def model_info_api(timeout=10):
    response = requests.get(
        f"{API_BASE_URL}/model-info",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def predict_api(payload, timeout=120):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def rag_api(
    question,
    n_results=4,
    timeout=900,
):
    response = requests.post(
        f"{API_BASE_URL}/rag",
        json={
            "question": question,
            "n_results": n_results,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
