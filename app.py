
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import shap
import folium
import h3

from src.model_utils import get_map_context
from src.api_client import (
    health_api,
    predict_api,
    rag_api,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="VUT Málaga · Sistema inteligente",
        layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37,99,235,.08), transparent 28%),
                radial-gradient(circle at top right, rgba(14,165,233,.07), transparent 24%),
                #f7f9fc;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .hero {
            padding: 2.1rem 2.3rem;
            border-radius: 24px;
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
            box-shadow: 0 16px 38px rgba(15,23,42,.18);
            margin-bottom: 1.3rem;
        }

        .hero-eyebrow {
            color: #bae6fd;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: .11em;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }

        .hero-title {
            color: white;
            font-size: 2.15rem;
            line-height: 1.08;
            font-weight: 800;
            margin-bottom: .65rem;
        }

        .hero-copy {
            color: #dbeafe;
            font-size: 1.02rem;
            max-width: 850px;
            margin: 0;
        }

        .section-kicker {
            color: #2563eb;
            font-size: .78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .09em;
            margin-bottom: .2rem;
        }

        .result-card {
            border-radius: 22px;
            padding: 1.7rem 1.9rem;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 72%);
            border: 1px solid #bfdbfe;
            box-shadow: 0 12px 30px rgba(37,99,235,.10);
            margin: .4rem 0 1.2rem 0;
        }

        .result-label {
            color: #475569;
            font-weight: 650;
            font-size: .9rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .result-price {
            color: #0f172a;
            font-size: 3rem;
            line-height: 1;
            font-weight: 850;
            margin: .35rem 0;
        }

        .result-sub {
            color: #64748b;
            font-size: .92rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.84);
            border: 1px solid #e2e8f0;
            padding: .95rem 1rem;
            border-radius: 16px;
            box-shadow: 0 5px 16px rgba(15,23,42,.04);
        }

        div[data-testid="stForm"] {
            border: 1px solid #dbe4f0;
            background: rgba(255,255,255,.78);
            border-radius: 20px;
            padding: 1.1rem 1.1rem .6rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,.04);
        }

        .rag-answer {
            background: white;
            border: 1px solid #dbe4f0;
            border-left: 5px solid #2563eb;
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 6px 18px rgba(15,23,42,.05);
            margin-top: .6rem;
        }

        .mini-note {
            color: #64748b;
            font-size: .86rem;
        }

        .market-card {
            background: rgba(255,255,255,.88);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 16px rgba(15,23,42,.04);
            height: 100%;
        }

        .market-label {
            color: #64748b;
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: .3rem;
        }

        .market-value {
            color: #0f172a;
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .market-sub {
            color: #64748b;
            font-size: .82rem;
            margin-top: .3rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
            background: rgba(255,255,255,.72);
            padding: .35rem;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# HELPERS API
# ============================================================

def shap_from_api(shap_payload):
    return shap.Explanation(
        values=np.asarray(
            shap_payload["values"],
            dtype=float,
        ),
        base_values=float(
            shap_payload["base_value"]
        ),
        data=np.asarray(
            shap_payload["data"],
            dtype=object,
        ),
        feature_names=list(
            shap_payload["feature_names"]
        ),
    )


def api_status():
    try:
        result = health_api(timeout=3)
        return result.get("status") == "ok"
    except Exception:
        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## VUT Málaga")
    st.caption("Sistema inteligente de apoyo a la estimación y consulta normativa")

    st.divider()

    backend_ok = api_status()

    if backend_ok:
        st.success("API REST conectada")
    else:
        st.error("API REST no disponible")

    st.markdown("### Motor predictivo")
    st.markdown("**Modelo:** XGBoost")
    st.markdown("**Objetivo:** tarifa por noche")
    st.markdown("**Validación:** holdout espacial 80/20")
    st.markdown("**MAE test:** 33,20 €")

    st.divider()

    st.markdown("### Motor documental")
    st.markdown("**Retrieval:** ChromaDB + embeddings")
    st.markdown("**LLM local:** Ollama")
    st.markdown("**Ámbito:** VUT Málaga / Andalucía")

    st.divider()
    st.caption(
        "TFM · Sistema predictivo, geoespacial y conversacional "
        "para viviendas de uso turístico."
    )


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Trabajo Fin de Máster · Málaga</div>
        <div class="hero-title">Sistema inteligente para viviendas de uso turístico</div>
        <p class="hero-copy">
            Estimación de tarifa mediante Machine Learning, análisis geoespacial
            e interpretación SHAP, junto con un consultor documental basado en RAG.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs([
    "Estimador de tarifa",
    "Consultor normativo",
])

if not api_status():
    st.warning(
        "La interfaz está disponible, pero la API REST no responde. "
        "Arranca Uvicorn en el puerto 8000 para habilitar predicciones y consultas RAG."
    )


# ============================================================
# PANEL 1
# ============================================================

with tab1:

    left_intro, right_intro = st.columns([2.2, 1])

    with left_intro:
        st.markdown(
            '<div class="section-kicker">Motor predictivo</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Configura el alojamiento")
        st.write(
            "Introduce los datos básicos de la vivienda. El sistema completa "
            "automáticamente las variables históricas y geoespaciales necesarias "
            "para generar la estimación."
        )

    with right_intro:
        st.info(
            "La tarifa es una estimación orientativa basada en patrones históricos. "
            "No representa una recomendación comercial vinculante."
        )

    with st.form("prediction_form"):

        st.markdown("#### 📍 Ubicación")
        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitud",
                value=36.7200,
                format="%.6f",
                help="Coordenada de la vivienda dentro del municipio de Málaga.",
            )

        with col2:
            longitude = st.number_input(
                "Longitud",
                value=-4.4200,
                format="%.6f",
            )

        st.markdown("#### 🏡 Características principales")
        col1, col2, col3 = st.columns(3)

        with col1:
            room_type = st.selectbox(
                "Tipo de alojamiento",
                [
                    "Entire home/apt",
                    "Private room",
                    "Shared room",
                    "Hotel room",
                ],
            )

            property_type_group = st.selectbox(
                "Tipo de propiedad",
                [
                    "Apartamento",
                    "Casa_Villa",
                    "Habitacion",
                    "Otros",
                ],
            )

        with col2:
            accommodates = st.number_input(
                "Capacidad máxima",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
            )

            bedrooms = st.number_input(
                "Dormitorios",
                min_value=0,
                max_value=15,
                value=2,
                step=1,
            )

        with col3:
            bathrooms_num = st.number_input(
                "Número de baños",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
            )

            host_is_superhost = st.checkbox(
                "⭐ El anfitrión es Superhost",
                value=False,
            )

        st.markdown("#### 📅 Condiciones de reserva")
        col1, col2, col3 = st.columns(3)

        with col1:
            minimum_nights = st.number_input(
                "Noches mínimas",
                min_value=1,
                max_value=30,
                value=2,
                step=1,
            )

        with col2:
            maximum_nights = st.number_input(
                "Noches máximas",
                min_value=1,
                max_value=1125,
                value=365,
                step=1,
            )

        with col3:
            instant_bookable = st.checkbox(
                "⚡ Reserva instantánea",
                value=True,
            )

        st.markdown("#### ✨ Equipamiento")

        amenity_labels = {
            "has_wifi": "📶 Wifi",
            "has_kitchen": "🍳 Cocina",
            "has_air_conditioning": "❄️ Aire acondicionado",
            "has_heating": "🔥 Calefacción",
            "has_parking": "🚗 Parking",
            "has_pool": "🏊 Piscina",
            "has_washer": "🧺 Lavadora",
            "has_dryer": "♨️ Secadora",
            "has_tv": "📺 Televisión",
            "has_balcony_or_terrace": "🌿 Balcón o terraza",
            "has_sea_view": "🌊 Vistas al mar",
            "has_workspace": "💻 Zona de trabajo",
            "has_elevator": "🛗 Ascensor",
            "has_pets_allowed": "🐾 Mascotas",
            "has_crib": "👶 Cuna",
            "has_bbq": "🔥 Barbacoa",
            "has_gym": "🏋️ Gimnasio",
            "has_hot_tub": "🫧 Jacuzzi",
            "has_breakfast": "☕ Desayuno",
        }

        selected_amenities = []
        amenity_columns = st.columns(4)

        for i, (amenity, label) in enumerate(
            amenity_labels.items()
        ):
            with amenity_columns[i % 4]:
                checked = st.checkbox(
                    label,
                    key=f"amenity_{amenity}",
                )
                if checked:
                    selected_amenities.append(amenity)

        submitted = st.form_submit_button(
            "Calcular tarifa estimada",
            use_container_width=True,
            type="primary",
        )

    # Calcular solo cuando se envía el formulario.
    # Cambiar el radio del mapa no vuelve a ejecutar XGBoost ni SHAP.
    if submitted:
        try:
            with st.spinner(
                "Enviando datos a la API y estimando tarifa..."
            ):
                payload = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "room_type": room_type,
                    "property_type_group": property_type_group,
                    "accommodates": accommodates,
                    "bedrooms": bedrooms,
                    "bathrooms_num": bathrooms_num,
                    "minimum_nights": minimum_nights,
                    "maximum_nights": maximum_nights,
                    "instant_bookable": int(instant_bookable),
                    "host_is_superhost": int(host_is_superhost),
                    "selected_amenities": selected_amenities,
                }

                api_result = predict_api(
                    payload,
                    timeout=120,
                )

                result = {
                    "price": float(
                        api_result["price"]
                    ),
                    "features": api_result["features"],
                    "shap_explanation": shap_from_api(
                        api_result["shap"]
                    ),
                }

            st.session_state["prediction_result"] = result
            st.session_state["prediction_coordinates"] = {
                "latitude": latitude,
                "longitude": longitude,
            }

        except Exception as error:
            st.error(
                f"No se ha podido realizar la predicción: {error}"
            )

    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]
        coordinates = st.session_state["prediction_coordinates"]

        result_latitude = coordinates["latitude"]
        result_longitude = coordinates["longitude"]
        features = result["features"]

        st.divider()

        st.markdown(
            '<div class="section-kicker">Resultado del modelo</div>',
            unsafe_allow_html=True,
        )

        result_col, context_col = st.columns([1.15, 1.85])

        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Tarifa estimada</div>
                    <div class="result-price">{result['price']:.2f} €</div>
                    <div class="result-sub">por noche · estimación XGBoost</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with context_col:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Distrito",
                    str(features["district"]),
                )

            with c2:
                st.metric(
                    "Playa más cercana",
                    f"{features['distance_nearest_beach_km']:.2f} km",
                )

            with c3:
                st.metric(
                    "Centro histórico",
                    f"{features['distance_nearest_historic_center_km']:.2f} km",
                )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Competidores H3",
                    f"{int(features['h3_res8_competitor_count']):,}",
                )

            with c2:
                st.metric(
                    "Densidad H3",
                    f"{features['h3_res8_density_km2']:.1f} aloj./km²",
                )

        # ========================================================
        # MAPA
        # ========================================================

        st.markdown(
            '<div class="section-kicker">Análisis geoespacial</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Mapa de competencia y proximidad")

        map_control, map_info = st.columns([1.2, 2.8])

        with map_control:
            radius_m = st.select_slider(
                "Radio de análisis",
                options=[500, 1000, 2000],
                value=1000,
                format_func=lambda x: (
                    f"{x} m"
                    if x < 1000
                    else f"{x / 1000:.0f} km"
                ),
            )

        radius_km = radius_m / 1000

        map_context = get_map_context(
            latitude=result_latitude,
            longitude=result_longitude,
            radius_km=radius_km,
        )

        competitors = map_context["competitors"]
        nearby = map_context["nearby"]

        with map_info:
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "En la misma celda H3",
                    f"{len(competitors):,}",
                )
            with m2:
                st.metric(
                    f"A ≤ {radius_m} m",
                    f"{len(nearby):,}",
                )

        # ========================================================
        # COMPARACIÓN CON EL MERCADO CERCANO
        # ========================================================

        st.markdown("#### Comparación con el mercado cercano")

        nearby_prices = (
            nearby["price"]
            .dropna()
            .astype(float)
        )

        if len(nearby_prices) > 0:
            nearby_mean = float(
                nearby_prices.mean()
            )

            nearby_median = float(
                nearby_prices.median()
            )

            price_percentile = float(
                (
                    nearby_prices
                    <= float(result["price"])
                ).mean()
                * 100
            )

            median_difference_pct = (
                (
                    float(result["price"])
                    - nearby_median
                )
                / nearby_median
                * 100
                if nearby_median > 0
                else np.nan
            )

            if np.isnan(median_difference_pct):
                comparison_text = "Sin referencia comparable"
            elif median_difference_pct >= 0:
                comparison_text = (
                    f"{median_difference_pct:.1f}% "
                    "sobre la mediana"
                )
            else:
                comparison_text = (
                    f"{abs(median_difference_pct):.1f}% "
                    "por debajo de la mediana"
                )

            mc1, mc2, mc3, mc4 = st.columns(4)

            with mc1:
                st.markdown(
                    f'''
                    <div class="market-card">
                        <div class="market-label">Tarifa estimada</div>
                        <div class="market-value">{result["price"]:.2f} €</div>
                        <div class="market-sub">Modelo XGBoost</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            with mc2:
                st.markdown(
                    f'''
                    <div class="market-card">
                        <div class="market-label">Mediana del entorno</div>
                        <div class="market-value">{nearby_median:.2f} €</div>
                        <div class="market-sub">{comparison_text}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            with mc3:
                st.markdown(
                    f'''
                    <div class="market-card">
                        <div class="market-label">Media del entorno</div>
                        <div class="market-value">{nearby_mean:.2f} €</div>
                        <div class="market-sub">{len(nearby_prices):,} alojamientos comparables</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

            with mc4:
                st.markdown(
                    f'''
                    <div class="market-card">
                        <div class="market-label">Percentil de precio</div>
                        <div class="market-value">P{price_percentile:.0f}</div>
                        <div class="market-sub">
                            Aproximadamente {price_percentile:.0f}% del entorno
                            tiene una tarifa igual o inferior
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

        else:
            st.info(
                "No hay suficientes alojamientos cercanos para "
                "calcular una comparación de mercado."
            )

        m = folium.Map(
            location=[result_latitude, result_longitude],
            zoom_start=14,
            tiles="CartoDB positron",
            control_scale=True,
        )

        district_gdf = map_context["district_gdf"]

        if not district_gdf.empty:
            district_layer = folium.FeatureGroup(
                name="Distrito",
                show=True,
            )

            folium.GeoJson(
                district_gdf.__geo_interface__,
                tooltip=folium.GeoJsonTooltip(
                    fields=["district"],
                    aliases=["Distrito:"],
                ),
            ).add_to(district_layer)

            district_layer.add_to(m)

        h3_group = folium.FeatureGroup(
            name="Celda H3 res. 8",
            show=True,
        )

        h3_cell = map_context["h3_res8"]
        h3_boundary = h3.cell_to_boundary(h3_cell)

        h3_polygon = [
            [lat, lon]
            for lat, lon in h3_boundary
        ]

        folium.Polygon(
            locations=h3_polygon,
            tooltip="Celda H3 res. 8",
            fill=True,
            fill_opacity=0.12,
            weight=3,
        ).add_to(h3_group)

        h3_group.add_to(m)

        competitors_group = folium.FeatureGroup(
            name="Competidores H3",
            show=True,
        )

        for _, competitor in competitors.iterrows():

            popup_text = (
                f"<b>Tarifa:</b> {competitor['price']:.2f} €<br>"
                f"<b>Tipo:</b> {competitor['room_type']}<br>"
                f"<b>Propiedad:</b> {competitor['property_type_group']}"
            )

            folium.CircleMarker(
                location=[
                    competitor["latitude"],
                    competitor["longitude"],
                ],
                radius=3,
                tooltip="Alojamiento competidor",
                popup=popup_text,
                fill=True,
                fill_opacity=0.60,
                weight=1,
            ).add_to(competitors_group)

        competitors_group.add_to(m)

        nearby_group = folium.FeatureGroup(
            name=f"Alojamientos a ≤ {radius_m} m",
            show=True,
        )

        folium.Circle(
            location=[
                result_latitude,
                result_longitude,
            ],
            radius=radius_m,
            tooltip=f"Radio de proximidad: {radius_m} m",
            fill=False,
            weight=2,
        ).add_to(nearby_group)

        for _, listing in nearby.iterrows():

            popup_nearby = (
                f"<b>Tarifa:</b> {listing['price']:.2f} €<br>"
                f"<b>Distancia:</b> {listing['distance_km']:.2f} km<br>"
                f"<b>Tipo:</b> {listing['room_type']}<br>"
                f"<b>Propiedad:</b> {listing['property_type_group']}"
            )

            folium.CircleMarker(
                location=[
                    listing["latitude"],
                    listing["longitude"],
                ],
                radius=2,
                tooltip=(
                    f"{listing['distance_km']:.2f} km · "
                    f"{listing['price']:.0f} €"
                ),
                popup=popup_nearby,
                fill=True,
                fill_opacity=0.32,
                weight=1,
            ).add_to(nearby_group)

        nearby_group.add_to(m)

        folium.Marker(
            location=[
                result_latitude,
                result_longitude,
            ],
            tooltip="Vivienda analizada",
            popup=(
                f"<b>Vivienda analizada</b><br>"
                f"Tarifa estimada: {result['price']:.2f} €"
            ),
            icon=folium.Icon(
                icon="home",
                prefix="fa",
            ),
        ).add_to(m)

        folium.LayerControl(
            position="topright",
            collapsed=True,
        ).add_to(m)

        map_html = m.get_root().render()

        components.html(
            map_html,
            height=610,
            scrolling=False,
        )

        st.caption(
            "Usa el selector de capas situado en la esquina superior derecha "
            "para mostrar u ocultar distrito, celda H3 y alojamientos."
        )

        # ========================================================
        # SHAP
        # ========================================================

        st.markdown(
            '<div class="section-kicker">Interpretabilidad</div>',
            unsafe_allow_html=True,
        )

        with st.expander(
            "¿Por qué el modelo estima esta tarifa?",
            expanded=True,
        ):
            st.write(
                "La interpretación se divide en dos vistas: primero se muestran "
                "los cinco factores con mayor impacto absoluto y después el "
                "waterfall completo de la predicción."
            )

            explanation = result["shap_explanation"]

            factor_names = np.asarray(
                explanation.feature_names,
                dtype=object,
            )

            factor_values = np.asarray(
                explanation.values,
                dtype=float,
            ).reshape(-1)

            # Comprobación defensiva por si SHAP devuelve una estructura inesperada.
            if factor_names.shape[0] != factor_values.shape[0]:
                st.warning(
                    "No se ha podido construir el resumen de factores SHAP "
                    "por una diferencia entre nombres y contribuciones."
                )

            else:
                top_factor_indices = np.argsort(
                    np.abs(factor_values)
                )[-5:]

                top_names = []
                top_values = []

                for factor_index in top_factor_indices:
                    clean_name = str(
                        factor_names[factor_index]
                    )

                    clean_name = (
                        clean_name
                        .replace("numeric__", "")
                        .replace("categorical__", "")
                        .replace("_", " ")
                    )

                    top_names.append(clean_name)
                    top_values.append(
                        float(
                            factor_values[factor_index]
                        )
                    )

                st.markdown("#### Factores con mayor impacto")

                fig_top, ax_top = plt.subplots(
                    figsize=(9, 4.6)
                )

                y_positions = np.arange(
                    len(top_names)
                )

                ax_top.barh(
                    y_positions,
                    top_values,
                )

                ax_top.set_yticks(
                    y_positions
                )

                ax_top.set_yticklabels(
                    top_names
                )

                ax_top.axvline(
                    0,
                    linewidth=1,
                )

                ax_top.set_xlabel(
                    "Impacto sobre la tarifa estimada (€)"
                )

                ax_top.set_ylabel("")

                ax_top.grid(
                    axis="x",
                    alpha=0.18,
                )

                for position, value in zip(
                    y_positions,
                    top_values,
                ):
                    offset = 0.35 if value >= 0 else -0.35
                    alignment = "left" if value >= 0 else "right"

                    ax_top.text(
                        value + offset,
                        position,
                        f"{value:+.2f} €",
                        va="center",
                        ha=alignment,
                        fontsize=9,
                    )

                fig_top.tight_layout()

                st.pyplot(
                    fig_top,
                    clear_figure=True,
                    use_container_width=True,
                )

                st.caption(
                    "Valores positivos elevan la tarifa estimada; "
                    "valores negativos la reducen."
                )

            st.markdown("#### Explicación completa")

            fig_waterfall = plt.figure(
                figsize=(10, 7)
            )

            shap.plots.waterfall(
                explanation,
                max_display=15,
                show=False,
            )

            plt.gcf().set_size_inches(
                10,
                7,
            )

            plt.subplots_adjust(
                top=0.93,
                left=0.30,
                right=0.95,
                bottom=0.08,
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True,
                use_container_width=True,
            )

            st.caption(
                "Las contribuciones SHAP están expresadas en la misma escala "
                "de salida del modelo y pueden interpretarse aproximadamente "
                "como euros añadidos o restados a la estimación."
            )


# ============================================================
# CALLBACK FAQ
# ============================================================

def load_faq_question(question):
    """
    Carga una pregunta frecuente en el cuadro de texto.

    Los callbacks de Streamlit se ejecutan antes de volver a renderizar
    los widgets, por lo que podemos actualizar rag_question sin provocar
    StreamlitAPIException ni lanzar automáticamente la consulta RAG.
    """
    st.session_state["rag_question"] = question


# ============================================================
# PANEL 2
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-kicker">Asistente documental</div>',
        unsafe_allow_html=True,
    )
    st.header("Consultor de normativa VUT")

    st.write(
        "Consulta la documentación incorporada al proyecto sobre "
        "viviendas de uso turístico en Málaga y Andalucía."
    )

    st.warning(
        "La respuesta se genera a partir del corpus documental del TFM. "
        "Es una herramienta informativa y no sustituye asesoramiento jurídico."
    )

    rag_main, rag_guide = st.columns(
        [1.85, 1.15],
        gap="large",
    )

    # --------------------------------------------------------
    # COLUMNA PRINCIPAL
    # --------------------------------------------------------

    with rag_main:

        st.markdown("### Realiza una consulta")

        rag_question = st.text_area(
            "Pregunta",
            placeholder=(
                "Escribe aquí una consulta sobre normativa, "
                "limitaciones urbanísticas, requisitos o funcionamiento "
                "de las viviendas de uso turístico..."
            ),
            height=135,
            key="rag_question",
        )

        rag_submit = st.button(
            "Consultar documentación",
            type="primary",
            key="rag_submit",
            use_container_width=True,
        )

        st.caption(
            "También puedes seleccionar una pregunta frecuente del panel derecho."
        )

        if rag_submit:

            if not rag_question.strip():
                st.warning(
                    "Escribe una pregunta antes de realizar la consulta."
                )

            else:
                with st.spinner(
                    "Consultando la API documental y generando respuesta..."
                ):
                    try:
                        api_rag_result = rag_api(
                            rag_question.strip(),
                            n_results=4,
                            timeout=900,
                        )

                        rag_result = {
                            "answer": api_rag_result["answer"],
                            "sources": api_rag_result["sources"],
                        }

                        st.session_state["rag_result"] = rag_result
                        st.session_state["rag_last_question"] = (
                            rag_question.strip()
                        )

                    except Exception as exc:
                        st.error(
                            f"Error al consultar la API RAG: {exc}"
                        )

        if "rag_result" in st.session_state:

            rag_result = st.session_state["rag_result"]

            st.divider()

            st.markdown(
                '<div class="section-kicker">Respuesta generada</div>',
                unsafe_allow_html=True,
            )

            st.subheader(
                st.session_state.get(
                    "rag_last_question",
                    "Respuesta",
                )
            )

            with st.container(border=True):
                st.markdown(
                    rag_result["answer"]
                )

            sources = rag_result["sources"]

            if sources:

                unique_sources = []
                seen_sources = set()

                for source in sources:
                    key = (
                        source.get("documento"),
                        source.get("pagina"),
                    )

                    if key not in seen_sources:
                        seen_sources.add(key)
                        unique_sources.append(source)

                with st.expander(
                    "Documentos consultados",
                    expanded=True,
                ):
                    for index, source in enumerate(
                        unique_sources,
                        start=1,
                    ):
                        st.markdown(
                            f"**{index}. {source.get('documento')}**  \n"
                            f"Página {source.get('pagina')}"
                        )

    # --------------------------------------------------------
    # PANEL DERECHO DE CONSULTAS GUIADAS
    # --------------------------------------------------------

    with rag_guide:

        st.markdown("### Preguntas frecuentes")
        st.caption(
            "Selecciona una consulta para cargarla automáticamente "
            "en el cuadro de pregunta."
        )

        faq_groups = {
            "Requisitos de la vivienda": [
                "¿Qué requisitos debe cumplir una vivienda de uso turístico?",
                "¿Qué condiciones debe cumplir la vivienda para poder destinarse a uso turístico?",
                "¿Qué obligaciones tiene la persona titular de una vivienda de uso turístico?",
                "¿Qué requisitos de equipamiento se exigen a una vivienda de uso turístico?",
            ],
            "Urbanismo y limitaciones": [
                "¿Qué ocurre cuando un barrio supera el 8% de viviendas de uso turístico?",
                "¿Existen zonas de Málaga donde se limite la implantación de nuevas viviendas de uso turístico?",
                "¿Cómo afecta el uso residencial del edificio a la implantación de una vivienda de uso turístico?",
                "¿Qué criterios utiliza el Ayuntamiento de Málaga para limitar las viviendas de uso turístico?",
            ],
            "Normativa aplicable": [
                "¿Qué establece la Instrucción 1/2024 sobre las viviendas de uso turístico?",
                "¿Qué regula el Decreto 28/2016 sobre viviendas de uso turístico?",
                "¿Qué cambios introduce el Decreto 31/2024?",
                "¿Qué acordó el Pleno del Ayuntamiento de Málaga el 29 de mayo de 2025 sobre las viviendas de uso turístico?",
            ],
            "Comunidad y compatibilidad de usos": [
                "¿Puede una comunidad de propietarios limitar o prohibir una vivienda de uso turístico?",
                "¿Puede existir una vivienda de uso turístico en un edificio de uso residencial?",
                "¿Cómo se relaciona la normativa turística con las limitaciones urbanísticas municipales?",
            ],
            "Control y cumplimiento": [
                "¿Qué puede ocurrir si una vivienda de uso turístico incumple la normativa?",
                "¿En qué casos puede dejar de considerarse válida una vivienda de uso turístico?",
                "¿Qué controles puede realizar la administración sobre las viviendas de uso turístico?",
            ],
        }

        for group_name, questions in faq_groups.items():

            with st.expander(
                group_name,
                expanded=(group_name == "Requisitos de la vivienda"),
            ):
                for i, question in enumerate(questions):
                    st.button(
                        question,
                        key=f"faq_{group_name}_{i}",
                        use_container_width=True,
                        on_click=load_faq_question,
                        args=(question,),
                    )

        st.markdown("---")
        st.caption(
            "Si ninguna de estas preguntas coincide con tu duda, "
            "puedes escribir una consulta libre en el panel principal."
        )
