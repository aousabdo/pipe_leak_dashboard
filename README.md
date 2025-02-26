# Pipe Leak Simulation and Prediction

This project simulates pipe leak events in a utility network and provides a predictive model to identify pipes at risk of future leaks. It includes a Streamlit dashboard for visualization and analysis.

## Features

- Simulation of realistic pipe network and leak events
- Machine learning model for leak prediction
- Interactive dashboard with maps, charts, and analysis tools
- GIS integration for spatial visualization

## Setup Instructions

1. **Create and activate a virtual environment**:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. **Install dependencies**:

```bash
# Install requirements
pip install -r requirements.txt
```

3. **Run the application**:

```bash
# Run the Streamlit app
streamlit run pipe_leak_app.py
```

## Project Structure

- `pipe_leak_simulation.py`: Generates synthetic pipe network and leak events data
- `pipe_leak_prediction.py`: Machine learning model for leak prediction
- `pipe_leak_app.py`: Streamlit dashboard application

## Data Generation

The first time you run the application, it will generate simulation data. You can regenerate data by checking the "Regenerate Simulation Data" option in the sidebar.

## Model Training

The application will train a machine learning model on the generated data. You can retrain the model by checking the "Retrain Prediction Model" option in the sidebar.

## Dashboard Features

- Historical leak visualization
- Risk prediction heatmap
- Leak pattern analysis
- Root cause analysis
- Model performance metrics

## Requirements

Python 3.9+ and the following packages:
- streamlit
- pandas
- numpy
- geopandas
- folium
- streamlit-folium
- matplotlib
- seaborn
- plotly
- scikit-learn
- xgboost
- imbalanced-learn
- shapely 