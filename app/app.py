import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, time

st.set_page_config(
    page_title="Dispatch Call Volume Forecaster",
    page_icon="📞",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/call_forecasting_model.pkl')
        model_cols = joblib.load('models/model_feature_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        return None, None

model, model_feature_columns = load_model()

def create_feature_vector(timestamp, census_data, feature_columns):
    """Creates a feature DataFrame for a single prediction in the correct order."""
    features_df = pd.DataFrame(columns=feature_columns)
    
    features_df.loc[0, 'hour_of_day'] = timestamp.hour
    features_df.loc[0, 'day_of_week'] = timestamp.weekday()
    features_df.loc[0, 'month'] = timestamp.month
    
    features_df.loc[0, 'rooms_with_patients'] = census_data['rooms_with_patients']
        
    return features_df[feature_columns]

st.title("📞 Dispatch Call Volume Forecaster")
st.markdown("This tool uses a machine learning model to predict the expected number of patient calls by category for a given hour.")

if model is None or model_feature_columns is None:
    st.error("Model not found. Please ensure `call_forecasting_model.pkl` and `model_feature_columns.pkl` are in the 'models' directory and that the training notebook has been run successfully.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 1: Enter Unit Status")
        organization_list = ['Unit 1', 'Unit 2', 'Unit 3', 'Floor 1', 'Floor 2', 'Floor 3']
        selected_org = st.selectbox("Select a Hospital Unit:", organization_list)
        
        rooms_with_patients = st.number_input(
            "Number of Rooms with Patients (Census):", 
            min_value=0, max_value=100, value=25, step=1,
            help="Enter the current number of occupied rooms on the unit."
        )

    with col2:
        st.subheader("Step 2: Select a Time to Forecast")
        now = datetime.now()
        next_hour = (now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1))
        
        forecast_date = st.date_input("Date:", value=next_hour.date())
        forecast_time_selection = st.time_input("Time (select the start of the hour):", value=next_hour.time(), step=3600)

        prediction_timestamp = datetime.combine(forecast_date, forecast_time_selection)

    st.divider()
    if st.button("Forecast Call Volume", type="primary", use_container_width=True):
        
        census_input = {'rooms_with_patients': rooms_with_patients}

        feature_vector = create_feature_vector(prediction_timestamp, census_input, model_feature_columns)
        
        prediction = model.predict(feature_vector)
        
        st.subheader(f"📈 Forecast for {selected_org} at {prediction_timestamp.strftime('%Y-%m-%d %H:%M')}")

        call_categories = ['Clinical', 'Mobility', 'Basic Need', 'Housekeeping', 'Personal Care', 'Other']
        
        total_predicted_calls = 0
        cols = st.columns(len(call_categories))
        for i, category in enumerate(call_categories):
            predicted_value = prediction[0][i]
            total_predicted_calls += predicted_value
            cols[i].metric(label=f"**{category}**", value=f"{predicted_value:.1f}")

        st.metric(label="**Total Predicted Calls**", value=f"{total_predicted_calls:.1f}")
        
        with st.expander("Show Raw Feature Vector Sent to Model"):
            st.write("The model made its prediction based on this input:")
            st.dataframe(feature_vector)