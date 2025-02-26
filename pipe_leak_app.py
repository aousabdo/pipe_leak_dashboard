import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipe_leak_simulation import PipeLeakSimulator
from pipe_leak_prediction import PipeLeakPredictor

# Set page configuration
st.set_page_config(
    page_title="PG&E Pipe Leak Dashboard",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
    }
    .section-header {
        font-size: 1.8rem;
        color: #336699;
        padding-top: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #4682B4;
        padding-top: 0.7rem;
    }
    .info-text {
        font-size: 1.1rem;
    }
    .metric-container {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #3366cc !important;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown('<div class="main-header">PG&E Pipe Leak Simulation & Prediction Dashboard</div>', unsafe_allow_html=True)

@st.cache_resource
def load_or_generate_data(regenerate=False, num_pipes=1000, timespan_days=365):
    """Load saved data or generate new simulation data"""
    if regenerate or not os.path.exists("pipe_network.csv") or not os.path.exists("leak_events.csv"):
        st.info("Generating new simulation data... This may take a moment.")
        simulator = PipeLeakSimulator(seed=42)
        pipe_network, leak_events = simulator.generate_simulation_data(num_pipes, timespan_days)
        pipe_network, leak_events = simulator.augment_data_with_environmental_factors(pipe_network, leak_events)
        
        # Save data
        pipe_network.drop(columns=['geometry']).to_csv("pipe_network.csv", index=False)
        
        # Check if leak_events is empty before saving
        if not leak_events.empty and 'geometry' in leak_events.columns:
            leak_events.drop(columns=['geometry']).to_csv("leak_events.csv", index=False)
        else:
            # Create an empty DataFrame with expected columns
            empty_leak_df = pd.DataFrame(columns=[
                'pipe_id', 'date', 'latitude', 'longitude', 'severity', 
                'flow_rate_gpm', 'detection_hours', 'water_loss_gallons', 
                'repair_cost', 'material', 'diameter', 'installation_year',
                'age', 'soil_type', 'pressure', 'depth', 'prev_repairs',
                'last_inspection', 'traffic_load', 'elevation', 
                'ground_movement_risk', 'water_proximity', 'temp_fluctuation'
            ])
            empty_leak_df.to_csv("leak_events.csv", index=False)
        
        # Create ML features
        if not leak_events.empty:
            features_df = simulator.create_feature_dataset_for_ml(pipe_network, leak_events)
            features_df.to_csv("ml_features.csv", index=False)
        else:
            # Create a minimal feature dataset with just pipe information
            features_df = pd.DataFrame(pipe_network.drop(columns=['geometry']))
            # Add target column (all zeros since no leaks)
            features_df['had_leak_recently'] = 0
            # Convert categorical variables
            features_df = pd.get_dummies(features_df, columns=['material', 'diameter', 'soil_type'])
            # Drop ID columns
            features_df = features_df.drop(columns=['pipe_id'])
            features_df.to_csv("ml_features.csv", index=False)
        
        return pipe_network, leak_events, features_df
    else:
        # Load saved data
        pipe_network = pd.read_csv("pipe_network.csv")
        leak_events = pd.read_csv("leak_events.csv")
        
        # Convert to GeoDataFrame
        pipe_network['geometry'] = gpd.points_from_xy(pipe_network.longitude, pipe_network.latitude)
        pipe_network = gpd.GeoDataFrame(pipe_network, geometry='geometry')
        
        if not leak_events.empty:
            leak_events['date'] = pd.to_datetime(leak_events['date'])
            leak_events['geometry'] = gpd.points_from_xy(leak_events.longitude, leak_events.latitude)
            leak_events = gpd.GeoDataFrame(leak_events, geometry='geometry')
        else:
            # Create an empty GeoDataFrame with the right structure
            leak_events = gpd.GeoDataFrame(geometry=[])
        
        # Load ML features
        features_df = pd.read_csv("ml_features.csv")
        
        return pipe_network, leak_events, features_df

@st.cache_resource
def load_or_train_model(features_df, retrain=False):
    """Load saved model or train a new one"""
    if retrain or not os.path.exists("pipe_leak_model.joblib"):
        st.info("Training new prediction model... This may take a moment.")
        predictor = PipeLeakPredictor(model_type='xgboost')
        predictor.train(features_df, optimize=False)
        predictor.save_model("pipe_leak_model.joblib")
        return predictor
    else:
        return PipeLeakPredictor.load_model("pipe_leak_model.joblib")

def create_folium_map(center_lat, center_lon, zoom_start=9):
    """Create a Folium map centered at the specified coordinates"""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, 
                  tiles='CartoDB positron')
    return m

def create_leak_map(leak_df, pipe_df):
    """Create a map showing leak locations"""
    # If no leak events, just show the pipe network
    if leak_df.empty:
        # Calculate center of the pipe network
        center_lat = pipe_df['latitude'].mean()
        center_lon = pipe_df['longitude'].mean()
        
        # Create the base map
        m = create_folium_map(center_lat, center_lon)
        
        # Add a note about no leak data
        folium.map.Marker(
            [center_lat, center_lon],
            icon=folium.DivIcon(
                icon_size=(250, 36),
                icon_anchor=(125, 18),
                html='<div style="font-size: 18px; font-weight: bold; color: red; background-color: white; padding: 5px; border-radius: 5px;">No leak events generated</div>'
            )
        ).add_to(m)
        
        return m
    
    # Calculate center of the data
    center_lat = leak_df['latitude'].mean()
    center_lon = leak_df['longitude'].mean()
    
    # Create the base map
    m = create_folium_map(center_lat, center_lon)
    
    # Add leak locations as a marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each leak, color-coded by severity
    severity_colors = {
        'Minor': 'green',
        'Moderate': 'orange',
        'Major': 'red',
        'Critical': 'darkred'
    }
    
    for idx, row in leak_df.iterrows():
        color = severity_colors.get(row['severity'], 'blue')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=f"Pipe ID: {row['pipe_id']}<br>"
                   f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                   f"Severity: {row['severity']}<br>"
                   f"Flow Rate: {row['flow_rate_gpm']} GPM<br>"
                   f"Water Loss: {int(row['water_loss_gallons']):,} gallons<br>"
                   f"Repair Cost: ${int(row['repair_cost']):,}"
        ).add_to(marker_cluster)
    
    # Add a heatmap layer
    heat_data = [[row['latitude'], row['longitude'], row['flow_rate_gpm']] 
                 for idx, row in leak_df.iterrows()]
    HeatMap(heat_data, radius=15, gradient={'0.2': 'blue', '0.5': 'lime', '0.8': 'yellow', '1.0': 'red'}).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_risk_heatmap(pipe_df, predictions, pred_probs):
    """Create a risk heatmap based on model predictions"""
    # Add predictions to pipe_df
    pipe_with_preds = pipe_df.copy()
    pipe_with_preds['leak_risk'] = pred_probs
    pipe_with_preds['predicted_leak'] = predictions
    
    # Calculate center of the data
    center_lat = pipe_with_preds['latitude'].mean()
    center_lon = pipe_with_preds['longitude'].mean()
    
    # Create the base map
    m = create_folium_map(center_lat, center_lon)

    # Check if our predictions are from a DummyModel (all 0s when no leaks)
    if len(set(predictions)) == 1 and predictions[0] == 0:
        # Special case for dummy model with no leak events
        # Add a note about no leak data
        folium.map.Marker(
            [center_lat, center_lon],
            icon=folium.DivIcon(
                icon_size=(300, 36),
                icon_anchor=(150, 18),
                html='<div style="font-size: 18px; font-weight: bold; color: red; background-color: white; padding: 5px; border-radius: 5px;">No leak events to build risk model</div>'
            )
        ).add_to(m)
        
        return m
    
    # Add a heatmap of risk probability
    heat_data = [[row['latitude'], row['longitude'], row['leak_risk']] 
                 for idx, row in pipe_with_preds.iterrows()]
    HeatMap(heat_data, radius=15, gradient={'0.2': 'blue', '0.5': 'lime', '0.8': 'yellow', '1.0': 'red'},
            min_opacity=0.3, blur=10).add_to(m)
    
    # Add high-risk points as markers
    high_risk = pipe_with_preds[pipe_with_preds['predicted_leak'] == 1]
    high_risk_cluster = MarkerCluster(name="High Risk Pipes").add_to(m)
    
    for idx, row in high_risk.iterrows():
        risk_pct = row['leak_risk'] * 100
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=min(row['leak_risk'] + 0.3, 0.9),
            tooltip=f"Pipe ID: {row['pipe_id']}<br>"
                   f"Risk Score: {risk_pct:.1f}%<br>"
                   f"Material: {row['material']}<br>"
                   f"Age: {row['age']} years<br>"
                   f"Last Inspection: {row['last_inspection']} days ago"
        ).add_to(high_risk_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def plot_leak_severity_distribution(leak_df):
    """Plot the distribution of leak severity"""
    if leak_df.empty:
        # Return empty figure with message
        fig = px.bar(
            pd.DataFrame({'Severity': ['No Data'], 'Count': [0]}),
            x='Severity', y='Count',
            title='Leak Severity Distribution (No Data Available)'
        )
        fig.update_layout(annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No leak events to display",
            showarrow=False,
            font=dict(size=16)
        )])
        return fig
        
    severity_counts = leak_df['severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    
    # Create order for severity levels
    severity_order = ['Minor', 'Moderate', 'Major', 'Critical']
    severity_counts['Severity'] = pd.Categorical(severity_counts['Severity'], 
                                                categories=severity_order, 
                                                ordered=True)
    severity_counts = severity_counts.sort_values('Severity')
    
    fig = px.bar(severity_counts, x='Severity', y='Count', 
                color='Severity',
                color_discrete_map={'Minor': 'green', 'Moderate': 'orange', 
                                    'Major': 'red', 'Critical': 'darkred'},
                title='Leak Severity Distribution')
    fig.update_layout(xaxis_title='Severity Level', yaxis_title='Number of Leaks')
    
    return fig

def plot_leaks_over_time(leak_df):
    """Plot leaks over time"""
    if leak_df.empty:
        # Return empty figure with message
        fig = px.line(
            pd.DataFrame({'Date': [datetime.now()], 'Count': [0], 'Severity': ['No Data']}),
            x='Date', y='Count',
            title='Leaks Over Time (No Data Available)'
        )
        fig.update_layout(annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No leak events to display",
            showarrow=False,
            font=dict(size=16)
        )])
        return fig
        
    # Aggregate leaks by date
    leak_df['date'] = pd.to_datetime(leak_df['date'])
    leaks_by_date = leak_df.groupby([pd.Grouper(key='date', freq='M'), 'severity']).size().reset_index()
    leaks_by_date.columns = ['Date', 'Severity', 'Count']
    
    # Create order for severity levels
    severity_order = ['Minor', 'Moderate', 'Major', 'Critical']
    leaks_by_date['Severity'] = pd.Categorical(leaks_by_date['Severity'], 
                                             categories=severity_order, 
                                             ordered=True)
    
    fig = px.line(leaks_by_date, x='Date', y='Count', color='Severity',
                 color_discrete_map={'Minor': 'green', 'Moderate': 'orange', 
                                    'Major': 'red', 'Critical': 'darkred'},
                 title='Leaks Over Time by Severity')
    fig.update_layout(xaxis_title='Date', yaxis_title='Number of Leaks')
    
    return fig

def plot_leak_causes(pipe_df, leak_df):
    """Plot correlation between pipe characteristics and leaks"""
    if leak_df.empty:
        # Return empty figures with messages
        fig_material = px.bar(
            pd.DataFrame({'material': ['No Data'], 'leak_rate': [0]}),
            x='material', y='leak_rate',
            title='Leak Rate by Pipe Material (No Data Available)'
        )
        fig_material.update_layout(annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No leak events to display",
            showarrow=False,
            font=dict(size=16)
        )])
        
        fig_age = px.bar(
            pd.DataFrame({'age_bin': ['No Data'], 'leak_rate': [0]}),
            x='age_bin', y='leak_rate',
            title='Leak Rate by Pipe Age (No Data Available)'
        )
        fig_age.update_layout(annotations=[dict(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No leak events to display",
            showarrow=False,
            font=dict(size=16)
        )])
        
        return fig_material, fig_age
        
    # Merge pipe and leak data
    pipe_leak_data = leak_df[['pipe_id', 'severity']].merge(pipe_df, on='pipe_id')
    
    # Leak rate by pipe material
    material_leaks = pipe_leak_data.groupby('material').size().reset_index(name='leak_count')
    material_total = pipe_df['material'].value_counts().reset_index()
    material_total.columns = ['material', 'total_count']
    
    material_rates = material_leaks.merge(material_total, on='material')
    material_rates['leak_rate'] = material_rates['leak_count'] / material_rates['total_count'] * 100
    
    fig_material = px.bar(material_rates.sort_values('leak_rate', ascending=False), 
                         x='material', y='leak_rate',
                         title='Leak Rate by Pipe Material (%)',
                         color='leak_rate',
                         color_continuous_scale='Viridis')
    fig_material.update_layout(xaxis_title='Pipe Material', yaxis_title='Leak Rate (%)')
    
    # Create age bins
    pipe_leak_data['age_bin'] = pd.cut(pipe_leak_data['age'], 
                                      bins=[0, 20, 40, 60, 80, 100, 200],
                                      labels=['0-20', '21-40', '41-60', '61-80', '81-100', '100+'])
    
    # Leak rate by age
    age_leaks = pipe_leak_data.groupby('age_bin').size().reset_index(name='leak_count')
    
    # Total pipes by age bin
    pipe_df['age_bin'] = pd.cut(pipe_df['age'], 
                               bins=[0, 20, 40, 60, 80, 100, 200],
                               labels=['0-20', '21-40', '41-60', '61-80', '81-100', '100+'])
    age_total = pipe_df['age_bin'].value_counts().reset_index()
    age_total.columns = ['age_bin', 'total_count']
    
    age_rates = age_leaks.merge(age_total, on='age_bin')
    age_rates['leak_rate'] = age_rates['leak_count'] / age_rates['total_count'] * 100
    
    fig_age = px.bar(age_rates, x='age_bin', y='leak_rate',
                    title='Leak Rate by Pipe Age (%)',
                    color='leak_rate',
                    color_continuous_scale='Viridis')
    fig_age.update_layout(xaxis_title='Pipe Age (years)', yaxis_title='Leak Rate (%)')
    
    return fig_material, fig_age

def plot_model_performance(predictor, features_df):
    """Plot model performance metrics"""
    # Prepare data for evaluation
    X_train, X_test, y_train, y_test = predictor._prepare_data(features_df)
    
    # Get evaluation metrics
    metrics = predictor.evaluate(X_test, y_test)
    
    # Plot metrics
    metrics_fig = go.Figure()
    
    # Check if we have comprehensive metrics or just accuracy (single class case)
    if 'precision' in metrics and 'recall' in metrics and 'f1' in metrics and 'auc' in metrics:
        # Multi-class case - we have all metrics
        metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['auc']]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    else:
        # Single class case - only accuracy is available
        metric_values = [metrics['accuracy']]
        metric_names = ['Accuracy']
    
    metrics_fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de'][:len(metric_names)]
    ))
    
    metrics_fig.update_layout(
        title='Model Performance Metrics',
        yaxis=dict(
            title='Score',
            range=[0, 1.05]
        ),
        xaxis_title='Metric',
        showlegend=False
    )
    
    # Check if we're dealing with a DummyModel (one class case)
    if hasattr(predictor.model, '__class__') and predictor.model.__class__.__name__ == 'DummyModel':
        # For DummyModel, feature importances might now use the structured format
        try:
            # Try using the structured format first
            importances = predictor.feature_importances['mean']
            feature_names = predictor.feature_importances['names']
            
            # Sort features alphabetically (since they have equal importance)
            indices = np.argsort([name for name in feature_names])
            
            # Plot top 10 features
            n_features = min(10, len(feature_names))
            selected_indices = indices[:n_features]
            
            importance_fig = go.Figure()
            
            importance_fig.add_trace(go.Bar(
                y=[feature_names[i] for i in selected_indices],
                x=importances[selected_indices],
                orientation='h',
                marker_color='#3366cc'
            ))
            
            importance_fig.update_layout(
                title='Feature Importances (Equal for Dummy Model)',
                xaxis_title='Importance',
                yaxis=dict(
                    title='Feature',
                    autorange='reversed'
                ),
                annotations=[dict(
                    x=0.5, y=-0.15,
                    xref="paper", yref="paper",
                    text="All features have equal importance in a dummy model (only one class in data)",
                    showarrow=False,
                    font=dict(size=12)
                )]
            )
        except (KeyError, TypeError):
            # Fall back to the old dictionary format if needed
            n_features = min(10, len(predictor.feature_names))
            selected_features = sorted(predictor.feature_names)[:n_features]
            importances = [predictor.feature_importances.get(name, 1.0/len(predictor.feature_names)) 
                          for name in selected_features]
            
            importance_fig = go.Figure()
            
            importance_fig.add_trace(go.Bar(
                y=selected_features,
                x=importances,
                orientation='h',
                marker_color='#3366cc'
            ))
            
            importance_fig.update_layout(
                title='Feature Importances (Equal for Dummy Model)',
                xaxis_title='Importance',
                yaxis=dict(
                    title='Feature',
                    autorange='reversed'
                ),
                annotations=[dict(
                    x=0.5, y=-0.15,
                    xref="paper", yref="paper",
                    text="All features have equal importance in a dummy model (only one class in data)",
                    showarrow=False,
                    font=dict(size=12)
                )]
            )
    else:
        # Standard model with structured feature importances
        try:
            # Get feature importances
            importances = predictor.feature_importances['mean']
            std = predictor.feature_importances['std']
            feature_names = predictor.feature_importances['names']
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            n_features = min(10, len(feature_names))
            top_indices = indices[:n_features]
            
            importance_fig = go.Figure()
            
            importance_fig.add_trace(go.Bar(
                y=[feature_names[i] for i in top_indices],
                x=importances[top_indices],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=std[top_indices]
                ),
                marker_color='#3366cc'
            ))
            
            importance_fig.update_layout(
                title='Top 10 Feature Importances',
                xaxis_title='Permutation Importance',
                yaxis=dict(
                    title='Feature',
                    autorange='reversed'
                )
            )
        except (KeyError, TypeError):
            # Fallback if feature_importances structure is different or missing
            # Create a simplified feature importance plot
            importance_fig = go.Figure()
            
            importance_fig.update_layout(
                title='Feature Importances Not Available',
                annotations=[dict(
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    text="Feature importance data is not available for this model",
                    showarrow=False,
                    font=dict(size=14)
                )]
            )
    
    return metrics_fig, importance_fig

def main():
    """Main function for the Streamlit app"""
    # Sidebar options
    st.sidebar.header("Simulation Controls")
    
    # Option to regenerate data
    regenerate_data = st.sidebar.checkbox("Regenerate Simulation Data", value=False)
    
    # Simulation parameters (only visible when regenerating)
    num_pipes = st.sidebar.slider("Number of Pipes", min_value=100, max_value=5000, value=1000, step=100,
                                disabled=not regenerate_data)
    timespan_days = st.sidebar.slider("Simulation Timespan (days)", min_value=30, max_value=730, value=365, step=30,
                                    disabled=not regenerate_data)
    
    # Option to retrain model
    retrain_model = st.sidebar.checkbox("Retrain Prediction Model", value=False)
    
    # Map display options
    st.sidebar.subheader("Map Display Options")
    show_heatmap = st.sidebar.checkbox("Show Heatmap Layer", value=True)
    show_markers = st.sidebar.checkbox("Show Marker Layer", value=True)
    
    # Filter options
    st.sidebar.subheader("Filters")
    
    # Load or generate data
    pipe_network, leak_events, features_df = load_or_generate_data(regenerate_data, num_pipes, timespan_days)
    
    # Load or train model
    predictor = load_or_train_model(features_df, retrain_model)
    
    # Make predictions on pipe network
    pipe_features = features_df.drop(columns=['had_leak_recently'])
    predictions, pred_probs = predictor.predict(pipe_features)
    
    # Handle filtering differently if leak_events is empty
    if leak_events.empty:
        st.sidebar.warning("No leak events were generated in the simulation.")
        filtered_leaks = leak_events  # Just use the empty DataFrame
        
        # Disable filter controls
        st.sidebar.text("Date Range (No data available)")
        st.sidebar.text("Leak Severity (No data available)")
        
        # Still allow material filtering from pipe_network for the risk map
        material_options = pipe_network['material'].unique().tolist()
        selected_materials = st.sidebar.multiselect(
            "Pipe Material",
            options=material_options,
            default=material_options
        )
    else:
        # Date range filter
        min_date = leak_events['date'].min().date()
        max_date = leak_events['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_leaks = leak_events[
                (leak_events['date'].dt.date >= start_date) & 
                (leak_events['date'].dt.date <= end_date)
            ]
        else:
            filtered_leaks = leak_events
        
        # Severity filter
        severity_options = leak_events['severity'].unique().tolist()
        selected_severities = st.sidebar.multiselect(
            "Leak Severity",
            options=severity_options,
            default=severity_options
        )
        
        if selected_severities:
            filtered_leaks = filtered_leaks[filtered_leaks['severity'].isin(selected_severities)]
        
        # Material filter
        material_options = leak_events['material'].unique().tolist()
        selected_materials = st.sidebar.multiselect(
            "Pipe Material",
            options=material_options,
            default=material_options
        )
        
        if selected_materials:
            filtered_leaks = filtered_leaks[filtered_leaks['material'].isin(selected_materials)]
    
    # Dashboard layout
    # Top row - Key metrics
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Pipes", f"{len(pipe_network):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Leaks", f"{len(leak_events):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        leak_rate = len(leak_events) / len(pipe_network) * 100
        st.metric("Annual Leak Rate", f"{leak_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Check if leak_events is empty before accessing repair_cost
        if leak_events.empty:
            total_cost = 0
        else:
            total_cost = leak_events['repair_cost'].sum()
        st.metric("Total Repair Cost", f"${total_cost:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Middle row - Maps
    st.markdown('<div class="section-header">Leak Maps</div>', unsafe_allow_html=True)
    
    map_tabs = st.tabs(["Historical Leaks", "Risk Prediction"])
    
    with map_tabs[0]:
        st.markdown('<div class="info-text">Map showing historical leak locations and severity.</div>', unsafe_allow_html=True)
        leak_map = create_leak_map(filtered_leaks, pipe_network)
        folium_static(leak_map, width=1200, height=500)
    
    with map_tabs[1]:
        st.markdown('<div class="info-text">Map showing predicted leak risk for the next 90 days.</div>', unsafe_allow_html=True)
        risk_map = create_risk_heatmap(pipe_network, predictions, pred_probs)
        folium_static(risk_map, width=1200, height=500)
    
    # Bottom row - Analysis tabs
    st.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)
    
    analysis_tabs = st.tabs(["Leak Patterns", "Root Causes", "Prediction Model"])
    
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_severity = plot_leak_severity_distribution(filtered_leaks)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            fig_time = plot_leaks_over_time(filtered_leaks)
            st.plotly_chart(fig_time, use_container_width=True)
    
    with analysis_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_material, fig_age = plot_leak_causes(pipe_network, leak_events)
            st.plotly_chart(fig_material, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_age, use_container_width=True)
    
    with analysis_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_fig, importance_fig = plot_model_performance(predictor, features_df)
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        with col2:
            st.plotly_chart(importance_fig, use_container_width=True)
    
    # Optional diagnostic section
    with st.expander("Diagnostic Information"):
        st.markdown("### Pipe Network Data Sample")
        st.dataframe(pipe_network.head())
        
        st.markdown("### Leak Events Data Sample")
        st.dataframe(filtered_leaks.head())
        
        st.markdown("### Model Features Data Sample")
        st.dataframe(features_df.head())

if __name__ == "__main__":
    main()