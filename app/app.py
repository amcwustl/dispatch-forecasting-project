# app/app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Dispatch Call Volume Forecaster",
    page_icon="📞",
    layout="centered"
)

# --- Load Model and Columns ---
# Use a try-except block to handle the first run when the model isn't trained yet
try:
    model = joblib.load('app/call_forecasting_model.pkl')
    model_feature_columns = joblib.load('app/model_feature_columns.pkl')
    st.session_state['model_loaded'] = True
except FileNotFoundError:
    st.error("Model not found. Please run the training notebook first.")
    st.session_state['model_loaded'] = False


# --- Helper Function ---
def create_features(timestamp, census_data):
    """Creates a feature DataFrame for a single prediction."""
    # Create a DataFrame with a single row
    features = pd.DataFrame(index=[0])
    
    # Time-based features
    features['hour_of_day'] = timestamp.hour
    features['day_of_week'] = timestamp.weekday() # Monday=0, Sunday=6
    features['is_weekend'] = 1 if features['day_of_week'][0] >= 5 else 0
    
    # Census features from user input
    for key, value in census_data.items():
        features[key] = value
        
    # Ensure all columns from training are present and in the correct order
    # This is crucial for the model to work correctly
    final_features = pd.DataFrame(columns=model_feature_columns)
    final_features = pd.concat([final_features, features], ignore_index=True).fillna(0)
    
    return final_features[model_feature_columns]


# --- Web App UI ---
st.title("📞 Dispatch Call Volume Forecaster")
st.markdown("This tool predicts the expected number of patient calls by category for a given hour.")

# Only show the prediction interface if the model was loaded successfully
if st.session_state['model_loaded']:
    # --- User Inputs ---
    st.header("Step 1: Enter Unit Status")

    # We can simulate a few organizations for the user to choose from
    # In a real app, this list would come from a database or config file
    organization_list = ['Anonymized Unit A', 'Anonymized Unit B', 'Anonymized Unit C']
    selected_org = st.selectbox("Select a Hospital Unit:", organization_list)

    rooms_with_patients = st.number_input("Number of Rooms with Patients (Census):", min_value=0, max_value=100, value=25, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        offline_rooms = st.number_input("Offline Rooms:", min_value=0, max_value=50, value=1, step=1)
        unplugged_rooms = st.number_input("Unplugged Rooms:", min_value=0, max_value=50, value=3, step=1)
    with col2:
        suspended_rooms = st.number_input("Suspended Rooms:", min_value=0, max_value=50, value=0, step=1)
        low_battery_rooms = st.number_input("Low Battery Rooms:", min_value=0, max_value=50, value=2, step=1)

    st.header("Step 2: Select a Time to Forecast")
    
    forecast_date = st.date_input("Date:", value=datetime.now())
    forecast_time = st.time_input("Time:", value=datetime.now())

    # Combine date and time for the prediction timestamp
    prediction_timestamp = datetime.combine(forecast_date, forecast_time)

    # --- Prediction Logic ---
    if st.button("Forecast Call Volume", type="primary"):
        # Package census data
        census_input = {
            'rooms_with_patients': rooms_with_patients,
            'suspended_rooms_with_patients': suspended_rooms,
            'offline_rooms_with_patients': offline_rooms,
            'unplugged_rooms_with_patients': unplugged_rooms,
            'low_battery_rooms_with_patients': low_battery_rooms
        }

        # Create the feature vector for the model
        feature_vector = create_features(prediction_timestamp, census_input)
        
        # Make a prediction
        prediction = model.predict(feature_vector)
        
        # --- Display Results ---
        st.header("📈 Forecast Results")
        st.write(f"Predicted call volumes for **{selected_org}** at **{prediction_timestamp.strftime('%Y-%m-%d %H:%M')}**:")

        # Get the call categories from your model's output
        # Assuming the model was trained on these categories in this order
        call_categories = ['Pain', 'Mobility', 'Basic Need', 'Housekeeping', 'Clinical', 'Other']
        
        # Display each prediction, rounding to a reasonable value
        results_df = pd.DataFrame({
            'Call Category': call_categories,
            'Predicted Count': [round(p, 1) for p in prediction[0]]
        })
        
        # Calculate and display total
        total_predicted_calls = results_df['Predicted Count'].sum()
        st.metric(label="**Total Predicted Calls**", value=f"{total_predicted_calls:.1f}")
        
        st.dataframe(results_df, use_container_width=True)
        
        with st.expander("Show Raw Feature Vector"):
            st.write("The following data was sent to the model for prediction:")
            st.dataframe(feature_vector)