import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dispatch Call Forecaster",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css');

    :root {
        --bg-color: #0E1117; --card-bg: #161B22; --primary-accent: #6366F1;
        --text-color: #FAFAFA; --border-color: #30363D;
    }
    .stApp { font-family: 'Inter', sans-serif; background-color: var(--bg-color); color: var(--text-color); }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3 { color: var(--text-color); font-weight: 600; }
    
    .stButton > button {
        background-color: var(--primary-accent); color: white; border-radius: 8px; border: none;
        padding: 0.75rem 1.5rem; font-weight: 600; transition: background-color 0.2s ease;
    }
    .stButton > button:hover { background-color: #4F46E5; }
    
    .stSlider [data-baseweb="slider"] > div:nth-child(3) {
        background: var(--primary-accent) !important;
    }

    [data-testid="stMetric"], .st-expander {
        background-color: var(--card-bg); border: 1px solid var(--border-color);
        border-radius: 12px; padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Data and Model Loading
ORGANIZATION_DATA = [
    {"organization_id": 80, "hospital_name": "County General", "unit_name": "Floor 1"},
    {"organization_id": 99, "hospital_name": "County General", "unit_name": "Floor 2"},
    {"organization_id": 74, "hospital_name": "County General", "unit_name": "Floor 3"},
    {"organization_id": 35, "hospital_name": "County General", "unit_name": "Floor 4"},
    {"organization_id": 73, "hospital_name": "County General", "unit_name": "Floor 5"},
    {"organization_id": 36, "hospital_name": "County General", "unit_name": "Floor 6"},
    {"organization_id": 93, "hospital_name": "County General", "unit_name": "Floor 7"},
    {"organization_id": 94, "hospital_name": "County General", "unit_name": "Floor 8"},
    {"organization_id": 81, "hospital_name": "County General", "unit_name": "Floor 9"},
    {"organization_id": 78, "hospital_name": "Seattle Grace", "unit_name": "Unit 1"},
    {"organization_id": 91, "hospital_name": "Seattle Grace", "unit_name": "Unit 2"},
    {"organization_id": 71, "hospital_name": "Seattle Grace", "unit_name": "Unit 3"},
    {"organization_id": 90, "hospital_name": "Seattle Grace", "unit_name": "Unit 4"},
    {"organization_id": 70, "hospital_name": "Seattle Grace", "unit_name": "Unit 5"},
    {"organization_id": 72, "hospital_name": "Seattle Grace", "unit_name": "Unit 6"},
    {"organization_id": 82, "hospital_name": "Seattle Grace", "unit_name": "Unit 7"},
    {"organization_id": 77, "hospital_name": "Seattle Grace", "unit_name": "Unit 8"},
    {"organization_id": 89, "hospital_name": "Seattle Grace", "unit_name": "Unit 9"}
]

@st.cache_data
def load_organization_data():
    df = pd.DataFrame(ORGANIZATION_DATA)
    unit_to_org_id = pd.Series(df.organization_id.values, index=df.unit_name).to_dict()
    return df, unit_to_org_id

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/call_forecasting_model.pkl')
        model_cols = joblib.load('models/model_feature_columns.pkl')
        return model, model_cols
    except FileNotFoundError:
        return None, None

org_data, unit_to_org_id_map = load_organization_data()
model, model_feature_columns = load_model()

def generate_single_unit_forecast(start_time, duration, census, organization_id, feature_columns):
    timestamps = [start_time + timedelta(hours=h) for h in range(duration)]
    predictions = []
    
    org_id_cols = [col for col in feature_columns if col.startswith('organization_id_')]
    
    for ts in timestamps:
        feature_data = {
            'rooms_with_patients': census,
            'hour_of_day': ts.hour,
            'day_of_week': ts.weekday()
        }
        
        for col in org_id_cols:
            feature_data[col] = 0
        
        current_org_col = f'organization_id_{organization_id}'
        if current_org_col in feature_data:
            feature_data[current_org_col] = 1

        features_df = pd.DataFrame([feature_data])
        
        features_df = features_df[feature_columns]
        
        prediction = model.predict(features_df)
        
        positive_prediction = [max(0, val) for val in prediction[0]]
        predictions.append(positive_prediction)
        
    categories = ['Clinical', 'Mobility', 'Basic Need', 'Housekeeping', 'Other']
    return pd.DataFrame(predictions, columns=categories, index=pd.to_datetime(timestamps))

# Streamlit UI
st.title("Dispatch Call Volume Forecaster")

if model is None or org_data.empty:
    st.error("Model or organization data not found.")
else:
    st.markdown('<h3><i class="fa-solid fa-sitemap"></i> Step 1: Select Scope</h3>', unsafe_allow_html=True)
    scope_cols = st.columns(2)
    with scope_cols[0]:
        st.markdown("**Hospital**")
        selected_hospital = st.selectbox("Hospital", org_data['hospital_name'].unique(), label_visibility="collapsed", help="Select the hospital you wish to forecast for.")
    with scope_cols[1]:
        st.markdown("**Unit / Floor**")
        units_for_hospital = org_data[org_data['hospital_name'] == selected_hospital]['unit_name'].tolist()
        selected_unit = st.selectbox("Unit/Floor", ["All"] + units_for_hospital, label_visibility="collapsed", help="Select a specific unit or choose 'All' to forecast for the entire hospital.")

    st.markdown('<h3><i class="fa-solid fa-users"></i> Step 2: Set Patient Census</h3>', unsafe_allow_html=True)
    unit_census_map = {}
    if selected_unit == "All":
        census_cols = st.columns(min(len(units_for_hospital), 4))
        for i, unit in enumerate(units_for_hospital):
            census = census_cols[i % 4].number_input(f"{unit}", min_value=0, max_value=40, value=25, step=1, key=unit, help=f"Enter the current number of occupied rooms for {unit} (0-40).")
            unit_census_map[unit] = census
    else:
        census = st.slider(f"Patient Census for {selected_unit}:", 0, 50, 25, 1, help="Adjust the slider to the current number of occupied rooms for this unit.")
        unit_census_map[selected_unit] = census
    
    st.markdown('<h3><i class="fa-regular fa-clock"></i> Step 3: Define Time Range</h3>', unsafe_allow_html=True)
    time_cols = st.columns([1, 2])
    with time_cols[0]:
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start_date = st.date_input("Start Date", value=next_hour.date(), help="Choose the starting date for your forecast period.")
        start_time_val = st.time_input("Start Time", value=next_hour.time(), help="Choose the starting hour for your forecast period.")
        start_datetime = datetime.combine(start_date, start_time_val)
    with time_cols[1]:
        duration = st.slider("Forecast Duration (Hours)", 1, 24, 8, 1, help="Select the length of the forecast, from 1 to 24 hours.")

    st.divider()

    if st.button("Forecast Call Volume", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            unit_forecasts = {}
            for unit_name, census_val in unit_census_map.items():
                org_id = unit_to_org_id_map.get(unit_name)
                if org_id:
                    unit_forecasts[unit_name] = generate_single_unit_forecast(start_datetime, duration, census_val, org_id, model_feature_columns)
            
            st.session_state['unit_forecasts'] = unit_forecasts
            st.session_state['details'] = {"hospital": selected_hospital, "unit": selected_unit}
            st.success("Forecast generated!")

    if 'unit_forecasts' in st.session_state:
        st.header(f"📈 Forecast Results for {st.session_state['details']['hospital']}")
        
        unit_forecasts = st.session_state['unit_forecasts']
        all_units = list(unit_forecasts.keys())
        
        selected_units_for_view = all_units
        if st.session_state['details']['unit'] == "All":
            selected_units_for_view = st.multiselect("Filter units to display:", options=all_units, default=all_units, help="Select one or more units to view their combined forecast.")

        if not selected_units_for_view:
            st.warning("Please select at least one unit to display results.")
        else:
            valid_forecasts = [unit_forecasts[unit] for unit in selected_units_for_view if not unit_forecasts[unit].empty]
            if not valid_forecasts:
                st.warning("No valid forecast data to display for the selected units.")
            else:
                forecast_df = pd.concat(valid_forecasts).groupby(level=0).sum()
                
                total_calls = forecast_df.sum().sum()
                peak_hour = forecast_df.sum(axis=1).idxmax()
                peak_volume = forecast_df.sum(axis=1).max()
                metric_cols = st.columns(3)
                metric_cols[0].metric("Total Predicted Calls", f"{total_calls:.1f}")
                metric_cols[1].metric("Predicted Peak Hour", peak_hour.strftime('%H:%M'))
                metric_cols[2].metric("Calls During Peak Hour", f"{peak_volume:.1f}")
                
                st.divider()

                color_map = {'Clinical': '#6366F1', 'Mobility': '#14B8A6', 'Basic Need': '#F97316', 'Housekeeping': '#EC4899', 'Other': '#6B7280'}
                
                st.subheader("Hourly Call Volume by Category")
                fig_area = go.Figure()
                for category in forecast_df.columns:
                    fig_area.add_trace(go.Scatter(
                        x=forecast_df.index, y=forecast_df[category], mode='lines', stackgroup='one',
                        name=category, fillcolor=color_map.get(category), line=dict(width=0.5)
                    ))
                fig_area.update_layout(template="plotly_dark", hovermode='x unified', margin=dict(t=10, b=10), height=400,
                                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_area, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Total Call Distribution")
                    category_totals = forecast_df.sum().sort_values(ascending=False)
                    fig_pie = go.Figure(data=[go.Pie(labels=category_totals.index, values=category_totals.values, hole=0.4,
                                                     marker=dict(colors=[color_map.get(cat) for cat in category_totals.index]))])
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10), height=400, template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_right:
                    st.subheader("Detailed Forecast Data")
                    display_df = forecast_df.copy()
                    display_df['Total'] = display_df.sum(axis=1)
                    display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(display_df.style.format("{:.1f}"), height=400)

    st.divider()
    with st.expander("About This Application"):
        st.markdown("""
        This tool uses machine learning and historical anonymized call data to forecast call volume for the Dispatch Care application. 
        This app connects patients directly with their care providers via a hospital room tablet with specific options for assistance 
        linked directly to care provider phones. It is intended to provide hospital administrators a way to plan staff allocations 
        across their hospitals given the patient census, the day of the week, and the time range of the shift.

        Incoming calls on the system are categorized according to the descriptions below:
        """)
        st.dataframe(pd.DataFrame({
            "Category": ["Clinical", "Mobility", "Basic Need", "Housekeeping", "Other"],
            "Description": ["Medical issues, pain, medications", "Movement, bathroom, repositioning", "Food, water, meal requests", "Cleaning, linens, environment", "Administrative, personal, and miscellaneous"]
        }), use_container_width=True, hide_index=True)