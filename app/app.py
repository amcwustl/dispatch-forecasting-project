# app/app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, time, timedelta

st.set_page_config(
    page_title="Dispatch Call Volume Forecaster",
    page_icon="🏥",
    layout="wide"
)

# --- Data Structures and Model Loading ---

# The organization mapping data, generated from the cleaned notebook, is embedded here.
ORGANIZATION_DATA = [
    {
        "organization_id": 80,
        "hospital_name": "County General",
        "unit_name": "Floor 1"
    },
    {
        "organization_id": 99,
        "hospital_name": "County General",
        "unit_name": "Floor 2"
    },
    {
        "organization_id": 74,
        "hospital_name": "County General",
        "unit_name": "Floor 3"
    },
    {
        "organization_id": 35,
        "hospital_name": "County General",
        "unit_name": "Floor 4"
    },
    {
        "organization_id": 73,
        "hospital_name": "County General",
        "unit_name": "Floor 5"
    },
    {
        "organization_id": 36,
        "hospital_name": "County General",
        "unit_name": "Floor 6"
    },
    {
        "organization_id": 93,
        "hospital_name": "County General",
        "unit_name": "Floor 7"
    },
    {
        "organization_id": 94,
        "hospital_name": "County General",
        "unit_name": "Floor 8"
    },
    {
        "organization_id": 81,
        "hospital_name": "County General",
        "unit_name": "Floor 9"
    },
    {
        "organization_id": 78,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 1"
    },
    {
        "organization_id": 91,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 2"
    },
    {
        "organization_id": 71,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 3"
    },
    {
        "organization_id": 90,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 4"
    },
    {
        "organization_id": 70,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 5"
    },
    {
        "organization_id": 72,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 6"
    },
    {
        "organization_id": 82,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 7"
    },
    {
        "organization_id": 77,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 8"
    },
    {
        "organization_id": 89,
        "hospital_name": "Seattle Grace",
        "unit_name": "Unit 9"
    }
]


@st.cache_data
def load_organization_data():
    """Load the hospital and unit mapping from the embedded data."""
    df = pd.DataFrame(ORGANIZATION_DATA)
    return df

@st.cache_resource
def load_model():
    """Load the pre-trained model and feature columns."""
    try:
        model = joblib.load('models/call_forecasting_model.pkl')
        model_cols = joblib.load('models/model_feature_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        return None, None

org_data = load_organization_data()
model, model_feature_columns = load_model()

# --- Core Functions ---

def create_feature_vector(timestamp, census_data, feature_columns):
    """Creates a feature DataFrame for a single prediction in the correct order."""
    features_df = pd.DataFrame(columns=feature_columns)
    
    features_df.loc[0, 'hour_of_day'] = timestamp.hour
    features_df.loc[0, 'day_of_week'] = timestamp.weekday()
    features_df.loc[0, 'month'] = timestamp.month
    features_df.loc[0, 'rooms_with_patients'] = census_data
        
    return features_df[feature_columns]

def generate_hourly_forecast(start_time, duration_hours, unit_census_map, feature_columns):
    """Generates hourly predictions for a given time range and for one or more units."""
    predictions = []
    timestamps = []
    
    for hour in range(duration_hours):
        current_timestamp = start_time + timedelta(hours=hour)
        timestamps.append(current_timestamp)
        
        hourly_total_prediction = None
        
        for unit, census in unit_census_map.items():
            feature_vector = create_feature_vector(current_timestamp, census, feature_columns)
            prediction_array = model.predict(feature_vector)
            
            if hourly_total_prediction is None:
                hourly_total_prediction = prediction_array
            else:
                hourly_total_prediction += prediction_array
        
        predictions.append(hourly_total_prediction[0])
        
    call_categories = ['Clinical', 'Mobility', 'Basic Need', 'Housekeeping', 'Personal Care', 'Other']
    forecast_df = pd.DataFrame(predictions, columns=call_categories, index=pd.to_datetime(timestamps))
    
    return forecast_df

# --- Streamlit User Interface ---

st.title("🏥 Dispatch Call Volume Forecaster")
st.markdown("This tool uses a machine learning model to predict the expected number of patient calls by category for a given shift.")

if model is None or org_data is None:
    st.error("Model or organization data not found. Please ensure `call_forecasting_model.pkl` and `model_feature_columns.pkl` are in the 'models/' directory.")
else:
    # --- Input Section ---
    st.subheader("Step 1: Select Scope and Patient Census")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Dynamic Hierarchical Selectors from the embedded data
        hospital_list = org_data['hospital_name'].unique()
        selected_hospital = st.selectbox("Select a Hospital:", options=hospital_list)
        
        units_for_hospital = org_data[org_data['hospital_name'] == selected_hospital]['unit_name'].tolist()
        hospital_unit_options = ["All"] + units_for_hospital
        selected_unit = st.selectbox("Select a Unit / Floor (or 'All'):", options=hospital_unit_options)

    with col2:
        # Dynamic Census Input
        unit_census_map = {}
        if selected_unit == "All":
            st.markdown(f"**Enter the patient census for each unit at {selected_hospital}:**")
            # Use a flexible number of columns for census inputs
            census_cols = st.columns(len(units_for_hospital))
            for i, unit in enumerate(units_for_hospital):
                census = census_cols[i].number_input(f"{unit} Census", min_value=0, max_value=200, value=25, step=1, key=unit)
                unit_census_map[unit] = census
        else:
            census = st.slider(f"Number of Rooms with Patients on {selected_unit}:", min_value=0, max_value=50, value=25, step=1)
            unit_census_map[selected_unit] = census

    st.divider()

    st.subheader("Step 2: Select a Time Range to Forecast")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        forecast_start_date = st.date_input("Forecast Start Date:", value=next_hour.date())
        forecast_start_time = st.time_input("Forecast Start Time:", value=next_hour.time(), step=timedelta(hours=1))
        
        start_datetime = datetime.combine(forecast_start_date, forecast_start_time)

    with col4:
        duration = st.slider("Forecast Duration (Hours):", min_value=1, max_value=24, value=8, step=1, help="Select the number of hours to forecast, e.g., for a full shift.")

    st.divider()

    # --- Prediction and Output Section ---
    if st.button("Forecast Call Volume", type="primary", use_container_width=True):
        
        forecast_df = generate_hourly_forecast(start_datetime, duration, unit_census_map, model_feature_columns)
        
        end_time_str = (start_datetime + timedelta(hours=duration-1)).strftime('%H:%M')
        
        forecast_title = f"📈 Forecast for {selected_hospital}"
        if selected_unit != "All":
            forecast_title += f" ({selected_unit})"
            
        st.subheader(f"{forecast_title} from {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_time_str}")

        # Key Metrics
        total_predicted_calls = forecast_df.sum().sum()
        peak_hour = forecast_df.sum(axis=1).idxmax()
        peak_calls = forecast_df.sum(axis=1).max()

        metric_cols = st.columns(3)
        metric_cols[0].metric(label="**Total Predicted Calls for Shift**", value=f"{total_predicted_calls:.1f}")
        metric_cols[1].metric(label="**Predicted Peak Hour**", value=peak_hour.strftime('%H:%M'))
        metric_cols[2].metric(label="**Calls During Peak Hour**", value=f"{peak_calls:.1f}")
        
        st.markdown("---")

        # Time Series Chart
        st.write("#### Hourly Call Volume by Category")
        st.bar_chart(forecast_df)
        
        with st.expander("Show Raw Forecast Data and Model Inputs"):
            st.write("**Forecast Data Table:**")
            st.dataframe(forecast_df.style.format("{:.2f}"))
            
            st.write("**Model Inputs:**")
            st.json({
                "start_time": start_datetime.isoformat(),
                "duration_hours": duration,
                "unit_census_inputs": unit_census_map
            })