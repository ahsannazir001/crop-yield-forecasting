import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Yield Forecasting",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# API Configuration
# --------------------------------------------------

DEFAULT_API_URL = "https://crop-yield-forecasting-0j5q.onrender.com"

# --------------------------------------------------
# Session State: Store Historical Predictions
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# Dashboard Header
# --------------------------------------------------

st.title("🌱 Crop Yield Forecasting Dashboard")

st.markdown(
    """
    Predict crop yield using soil type, rainfall, temperature,
    and fertilizer inputs.

    This dashboard connects to the deployed FastAPI machine
    learning service.
    """
)

st.divider()

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------

st.sidebar.header("⚙️ API Configuration")

api_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=DEFAULT_API_URL
).strip().rstrip("/")

st.sidebar.markdown(
    f"[Open API Documentation]({api_url}/docs)"
)

if st.sidebar.button("🔍 Check API"):
    try:
        response = requests.get(
            api_url,
            timeout=30
        )

        if response.status_code == 200:
            st.sidebar.success("API is responding.")
        else:
            st.sidebar.warning(
                f"API returned HTTP {response.status_code}."
            )

    except requests.RequestException as error:
        st.sidebar.error(f"API connection failed: {error}")

# --------------------------------------------------
# Prediction Input Form
# --------------------------------------------------

st.subheader("📋 Enter Field Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        soil_type = st.selectbox(
            "Soil Type",
            options=["loamy", "clay", "sandy"]
        )

        rainfall = st.number_input(
            "Rainfall",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=10.0
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=70.0,
            value=25.0,
            step=1.0
        )

        fertilizer = st.number_input(
            "Fertilizer",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "🌾 Predict Crop Yield",
        use_container_width=True
    )

# --------------------------------------------------
# Call FastAPI Prediction Endpoint
# --------------------------------------------------

if submitted:

    if not api_url.startswith(("https://", "http://")):
        st.error("Please enter a valid HTTP or HTTPS API URL.")

    else:
        payload = {
            "soil_type": soil_type,
            "rainfall": rainfall,
            "temperature": temperature,
            "fertilizer": fertilizer
        }

        with st.spinner("Predicting crop yield..."):

            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    if "yield" not in result:
                        st.error(
                            "The API response does not contain "
                            "the expected 'yield' field."
                        )

                    else:
                        predicted_yield = float(result["yield"])

                        if not pd.notna(predicted_yield):
                            st.error(
                                "The API returned an invalid prediction."
                            )

                        else:
                            prediction_record = {
                                "Prediction": (
                                    len(
                                        st.session_state
                                        .prediction_history
                                    ) + 1
                                ),
                                "Predicted Yield": predicted_yield,
                                "Soil Type": soil_type,
                                "Rainfall": rainfall,
                                "Temperature": temperature,
                                "Fertilizer": fertilizer,
                                "Time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                            }

                            st.session_state.prediction_history.append(
                                prediction_record
                            )

                            st.success(
                                "Prediction generated successfully!"
                            )

                            st.subheader("📊 Prediction Result")

                            metric_col1, metric_col2 = st.columns(2)

                            with metric_col1:
                                st.metric(
                                    "Predicted Crop Yield",
                                    f"{predicted_yield:.3f}"
                                )

                            with metric_col2:
                                st.metric(
                                    "Measurement Unit",
                                    "Tonnes / Hectare"
                                )

                            st.caption(
                                "Prediction is based on the submitted "
                                "input features and the trained model."
                            )

                else:
                    st.error(
                        f"Prediction failed. "
                        f"HTTP Status: {response.status_code}"
                    )

                    st.code(response.text[:1000])

            except requests.Timeout:
                st.error(
                    "The API request timed out. "
                    "Please try again in a moment."
                )

            except requests.RequestException as error:
                st.error(f"Could not connect to the API: {error}")

            except (ValueError, TypeError):
                st.error(
                    "The API returned an invalid prediction value."
                )

# --------------------------------------------------
# Historical Predictions Chart
# --------------------------------------------------

st.divider()

st.subheader("📈 Historical Predictions")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    chart_df = history_df[
        ["Prediction", "Predicted Yield"]
    ].set_index("Prediction")

    st.bar_chart(
        chart_df,
        y="Predicted Yield"
    )

    st.subheader("📑 Prediction History")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No predictions yet. Enter field information and click "
        "'Predict Crop Yield' to generate your first prediction."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Crop Yield Forecasting and Alert System | "
    "Powered by Streamlit, FastAPI and Scikit-learn"
)