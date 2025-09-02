# Hospital Room Call Volume Forecaster

This project is a data-driven web application designed to forecast patient call volumes in a hospital setting. By leveraging machine learning, it provides unit managers with actionable, hour-by-hour predictions of both the number and type of patient calls (e.g., Clinical, Mobility, Basic Needs). This enables smarter, data-informed decisions about staff and resource allocation, helping to improve patient care and operational efficiency.

The application is built with Python, using Scikit-learn for the predictive model and Streamlit for the interactive user interface.

## Live Application

The application is deployed and publicly accessible via Streamlit Community Cloud.

**➡️ [Access the Live Forecaster Here](https://dispatch-call-volume-forecaster.streamlit.app/)**

## Features

- **Interactive Dashboard:** A user-friendly interface for selecting a hospital, unit, and patient census.
- **Hourly Forecasts:** Generates detailed predictions for a specified time range (1-24 hours).
- **Call Categorization:** Breaks down forecasts into five key categories: Clinical, Mobility, Basic Need, Housekeeping, and Other.
- **Rich Visualizations:** Presents data through a stacked area chart, a donut chart for call distribution, and a detailed data table.
- **Multi-Unit Analysis:** Supports forecasting for a single unit or an entire hospital by aggregating data from all its floors.

## Running the Application Locally

Follow these steps to set up and run the project on your own machine.

### Prerequisites

- Python (Version 3.8 or newer)
- `pip` (Python package installer)

### Step 1: Clone the Repository

First, clone the project repository or download a zip file of the project structure to your local machine:

```bash
git clone https://github.com/amcwustl/dispatch-forecasting-project.git
cd dispatch-forecasting-project
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:\nvenv\\Scripts\\activate
# On macOS/Linux:\nsource venv/bin/activate
```

### Step 3: Install Required Packages

With the virtual environment active, install all necessary libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Download and Place Data Files

The historical data files are required to run the model training notebook and are too large for the repository.

1. **Download the data zip file from this link:**  
    [**Dropbox: Download Data CSVs**](https://www.dropbox.com/scl/fo/5ltunxkhmirpy7scbte4k/AKFEs8ztRtEjO8CdBpMr7Yg?rlkey=i6qmikl530m197c4qo1mmaggq&st=jr5t87oc&dl=1)

2. **Unzip the file.** You will get two CSV files:
    - `call_created_anonymized.csv`
    - `census_check_anonymized.csv`

3. **Move these two CSV files** into the `data/` directory within the project folder.

Your `data` directory should now look like this:

```bash
dispatch-forecasting-project/
└── data/
    ├── call_created_anonymized.csv
    ├── census_check_anonymized.csv
    └── README.md
```

### Step 5: Launch the Streamlit Application

You are now ready to run the app. Execute the following command in your terminal from the project's root directory:

```bash
streamlit run app/app.py
```

Your default web browser will open a new tab with the application running locally.

## Optional: Retraining the Model

The pre-trained model is included in the `models/` directory. If you wish to retrain the model yourself or explore the data cleaning process, you can use the Jupyter Notebook.

1. Ensure you have completed all setup steps above, including downloading the data.
2. Launch Jupyter Lab from your terminal:

    ```bash
    jupyter lab
    ```

3. Navigate to the `notebooks/` directory and open `Data_Cleaning_and_Model_Training.ipynb`.
4. Run the cells sequentially to see the data processing steps and to generate new model files (`.pkl`) in the `models/` directory.

## Project Structure

```bash
├── app/
│   └── app.py                  # The main Streamlit application script.
├── data/
│   └── call_created_anonymized.csv
│   └── census_check_anonymized.csv
│   └── README.md               # Instructions for where to place the required CSV data files.
├── models/
│   ├── call_forecasting_model.pkl  # The serialized, pre-trained Random Forest model.
│   └── model_feature_columns.pkl # The list of feature columns the model was trained on.
├── notebooks/
│   └── Data_Cleaning_and_Model_Training.ipynb # Jupyter Notebook for data processing and model training.
├── README.md                   # This file.
└── requirements.txt            # A list of all Python packages required to run the project.
```
