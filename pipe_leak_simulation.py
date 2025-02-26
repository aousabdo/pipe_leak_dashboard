import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime
import random
import os
from sklearn.preprocessing import MinMaxScaler

class PipeLeakSimulator:
    """
    Simulator for generating realistic pipe leak data for utility networks
    """
    
    def __init__(self, seed=42):
        """Initialize the simulator with default parameters"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Load California shapefile
        shapefile_path = os.path.join("shape_files", "ca_state (1)", "CA_State.shp")
        try:
            self.ca_boundary = gpd.read_file(shapefile_path)
            print("California boundary shapefile loaded successfully")
            # Set CRS to WGS84 for consistency
            self.ca_boundary = self.ca_boundary.to_crs(epsg=4326)
            # Get the boundary geometry
            self.ca_geometry = self.ca_boundary.geometry.unary_union
        except Exception as e:
            print(f"Warning: Could not load California shapefile: {e}")
            self.ca_geometry = None
        
        # Define California boundaries (expanded to cover all of California)
        self.lat_bounds = (32.5, 42.0)  # latitude boundaries
        self.lon_bounds = (-124.5, -114.0)  # longitude boundaries
        
        # Pipe material types and their relative leak probabilities
        self.pipe_materials = {
            'Cast Iron': 0.35,  # highest risk, older material
            'Steel': 0.15,
            'PVC': 0.05,
            'HDPE': 0.03,  # lowest risk, modern material
            'Ductile Iron': 0.12,
            'Asbestos Cement': 0.30,
        }
        
        # Pipe diameter ranges (in inches) and their relative leak probabilities
        self.pipe_diameters = {
            'small (2-6)': 0.45,
            'medium (8-12)': 0.30,
            'large (14-24)': 0.15,
            'very large (>24)': 0.10,
        }
        
        # Soil types and their corrosion/leak factors
        self.soil_types = {
            'Clay': 0.23,
            'Sandy': 0.17,
            'Loam': 0.12,
            'Rocky': 0.28,
            'Silt': 0.20,
        }
        
        # Leak severity categories
        self.leak_severity = ['Minor', 'Moderate', 'Major', 'Critical']
        self.leak_severity_probs = [0.60, 0.25, 0.10, 0.05]  # probabilities for each severity
        
    def _generate_pipe_network(self, num_pipes=1000):
        """Generate a synthetic pipe network within California boundaries"""
        pipes = []
        pipe_count = 0
        
        # Max attempts to prevent infinite loop
        max_attempts = num_pipes * 10
        attempts = 0
        
        while pipe_count < num_pipes and attempts < max_attempts:
            attempts += 1
            
            # Generate random location within bounds
            latitude = np.random.uniform(self.lat_bounds[0], self.lat_bounds[1])
            longitude = np.random.uniform(self.lon_bounds[0], self.lon_bounds[1])
            
            # Create a point geometry
            point = Point(longitude, latitude)
            
            # Check if point is within California boundary
            if self.ca_geometry is not None and not self.ca_geometry.contains(point):
                continue  # Skip if point is outside California
            
            # Generate pipe characteristics
            material = np.random.choice(list(self.pipe_materials.keys()), 
                                        p=list(self.pipe_materials.values()))
            diameter = np.random.choice(list(self.pipe_diameters.keys()), 
                                       p=list(self.pipe_diameters.values()))
            
            # Installation year - older pipes are more likely to leak
            # Distribution skewed toward older pipes for realism
            installation_year = int(np.random.beta(2, 5) * (2023 - 1920) + 1920)
            
            # Calculate pipe age
            age = 2023 - installation_year
            
            # Soil type for this pipe location
            soil_type = np.random.choice(list(self.soil_types.keys()), 
                                        p=list(self.soil_types.values()))
            
            # Operating pressure (PSI)
            pressure = np.random.gamma(shape=5, scale=10)
            
            # Depth of pipe (feet)
            depth = np.random.uniform(3, 8)
            
            # Number of previous repairs (more common in older pipes)
            prev_repairs = np.random.poisson(max(0.1, age / 20))
            
            # Last inspection (days ago)
            last_inspection = np.random.randint(0, 365 * 3)  # up to 3 years
            
            # Traffic load above pipe (scale 1-10)
            traffic_load = np.random.randint(1, 11)
            
            pipes.append({
                'pipe_id': f'P{pipe_count:04d}',
                'latitude': latitude,
                'longitude': longitude,
                'material': material,
                'diameter': diameter,
                'installation_year': installation_year,
                'age': age,
                'soil_type': soil_type,
                'pressure': round(pressure, 2),
                'depth': round(depth, 2),
                'prev_repairs': prev_repairs,
                'last_inspection': last_inspection,
                'traffic_load': traffic_load,
                'geometry': point
            })
            
            pipe_count += 1
            
        if attempts >= max_attempts and pipe_count < num_pipes:
            print(f"Warning: Only able to generate {pipe_count} pipes within California boundary after {max_attempts} attempts")
        
        # Create GeoDataFrame with pipe network
        pipe_gdf = gpd.GeoDataFrame(pipes, geometry='geometry')
        return pipe_gdf
    
    def _calculate_leak_probability(self, pipe):
        """Calculate leak probability for a given pipe based on its characteristics"""
        # Base probability factors
        material_factor = {
            'Cast Iron': 0.015,
            'Steel': 0.010,
            'PVC': 0.005,
            'HDPE': 0.002,
            'Ductile Iron': 0.008,
            'Asbestos Cement': 0.014
        }
        
        diameter_factor = {
            'small (2-6)': 0.012,
            'medium (8-12)': 0.008, 
            'large (14-24)': 0.005,
            'very large (>24)': 0.003
        }
        
        soil_factor = {
            'Clay': 0.010,
            'Sandy': 0.008,
            'Loam': 0.005,
            'Rocky': 0.012,
            'Silt': 0.009
        }
        
        # Calculate base probability
        base_prob = material_factor[pipe['material']]
        
        # Apply modifiers
        # Age is a critical factor - exponential relationship with leak probability
        # Increase the exponent from 1.5 to 2.5 to make age have more impact
        age_factor = 1.0 + (pipe['age'] / 25) ** 2.5
        
        # Diameter modifier
        diameter_mod = diameter_factor[pipe['diameter']]
        
        # Soil type modifier
        soil_mod = soil_factor[pipe['soil_type']]
        
        # Pressure modifier (higher pressure = higher risk)
        pressure_mod = 1.0 + (pipe['pressure'] / 100)
        
        # Previous repairs (indicator of problematic pipe)
        repair_mod = 1.0 + (pipe['prev_repairs'] * 0.2)
        
        # Inspection recency (recently inspected pipes are safer)
        inspection_mod = 1.0 + (pipe['last_inspection'] / 1000)
        
        # Traffic load (more traffic = more ground stress)
        traffic_mod = 1.0 + (pipe['traffic_load'] / 20)
        
        # Calculate final probability
        leak_prob = base_prob * age_factor * diameter_mod * soil_mod * pressure_mod * repair_mod * inspection_mod * traffic_mod
        
        # Add a minimum probability to ensure some leaks are generated
        # This guarantees approximately 5-10% of pipes will have a leak during the year
        min_annual_probability = 0.05
        
        # Make the minimum probability depend on age to ensure older pipes have higher leak rates
        if pipe['age'] > 60:
            min_annual_probability = 0.10  # 10% minimum for very old pipes
        elif pipe['age'] > 40:
            min_annual_probability = 0.075  # 7.5% minimum for old pipes
        
        annual_probability = max(leak_prob, min_annual_probability)
        
        # Cap probability at 0.95 to avoid certainty
        return min(annual_probability, 0.95)
    
    def _generate_leak_events(self, pipe_network, timespan_days=365, seasonal_factor=True):
        """Generate leak events for the pipe network over a given timespan"""
        leak_events = []
        current_date = datetime.datetime(2023, 1, 1)
        end_date = current_date + datetime.timedelta(days=timespan_days)
        
        # Monthly temperature variation (proxy for seasonal effects)
        # Higher temps can cause pipe expansion and increased leaks
        monthly_temp_factors = {
            1: 1.2,  # January - cold, pipe contraction
            2: 1.15,
            3: 1.0,
            4: 0.9,
            5: 0.85,
            6: 0.8,  # June - mild, less stress
            7: 0.9,
            8: 1.0,
            9: 1.1,
            10: 1.2,
            11: 1.3,
            12: 1.35  # December - cold, pipe contraction
        }
        
        # Iterate through each day in the timespan
        while current_date < end_date:
            month = current_date.month
            temp_factor = monthly_temp_factors[month] if seasonal_factor else 1.0
            
            # For each pipe, check if a leak occurs on this day
            for idx, pipe in pipe_network.iterrows():
                daily_leak_prob = self._calculate_leak_probability(pipe) / 365 * temp_factor
                
                if np.random.random() < daily_leak_prob:
                    # Determine leak severity
                    severity = np.random.choice(self.leak_severity, p=self.leak_severity_probs)
                    
                    # Determine leak flow rate (gallons per minute) based on severity and pipe characteristics
                    if severity == 'Minor':
                        flow_rate = np.random.uniform(0.1, 5)
                    elif severity == 'Moderate':
                        flow_rate = np.random.uniform(5, 25)
                    elif severity == 'Major':
                        flow_rate = np.random.uniform(25, 100)
                    else:  # Critical
                        flow_rate = np.random.uniform(100, 500)
                    
                    # Adjust flow rate based on pipe diameter
                    if pipe['diameter'] == 'small (2-6)':
                        flow_multiplier = 0.7
                    elif pipe['diameter'] == 'medium (8-12)':
                        flow_multiplier = 1.0
                    elif pipe['diameter'] == 'large (14-24)':
                        flow_multiplier = 1.5
                    else:  # very large
                        flow_multiplier = 2.5
                    
                    actual_flow_rate = flow_rate * flow_multiplier
                    
                    # Calculate time to detection based on severity, inspection history
                    if severity == 'Critical':
                        detection_hours = np.random.uniform(0.5, 12)
                    elif severity == 'Major':
                        detection_hours = np.random.uniform(6, 48)
                    elif severity == 'Moderate':
                        detection_hours = np.random.uniform(24, 168)  # 1-7 days
                    else:  # Minor
                        detection_hours = np.random.uniform(72, 720)  # 3-30 days
                    
                    # Adjustment for inspection recency
                    if pipe['last_inspection'] < 30:  # inspected in last month
                        detection_hours *= 0.7
                    elif pipe['last_inspection'] > 365:  # not inspected in a year
                        detection_hours *= 1.5
                    
                    # Calculate water loss
                    water_loss_gallons = actual_flow_rate * 60 * detection_hours
                    
                    # Calculate repair cost based on severity, pipe characteristics
                    if severity == 'Minor':
                        base_repair_cost = np.random.uniform(1000, 5000)
                    elif severity == 'Moderate':
                        base_repair_cost = np.random.uniform(5000, 15000)
                    elif severity == 'Major':
                        base_repair_cost = np.random.uniform(15000, 50000)
                    else:  # Critical
                        base_repair_cost = np.random.uniform(50000, 250000)
                    
                    # Adjust for pipe diameter and depth
                    size_factor = 1.0
                    if pipe['diameter'] == 'medium (8-12)':
                        size_factor = 1.3
                    elif pipe['diameter'] == 'large (14-24)':
                        size_factor = 1.8
                    elif pipe['diameter'] == 'very large (>24)':
                        size_factor = 2.5
                    
                    depth_factor = 1.0 + (pipe['depth'] - 4) / 4
                    repair_cost = base_repair_cost * size_factor * depth_factor
                    
                    # Add leak event to the list
                    leak_events.append({
                        'pipe_id': pipe['pipe_id'],
                        'date': current_date,
                        'latitude': pipe['latitude'],
                        'longitude': pipe['longitude'],
                        'severity': severity,
                        'flow_rate_gpm': round(actual_flow_rate, 2),
                        'detection_hours': round(detection_hours, 1),
                        'water_loss_gallons': round(water_loss_gallons, 0),
                        'repair_cost': round(repair_cost, 2),
                        'material': pipe['material'],
                        'diameter': pipe['diameter'],
                        'installation_year': pipe['installation_year'],
                        'age': pipe['age'],
                        'soil_type': pipe['soil_type'],
                        'pressure': pipe['pressure'],
                        'depth': pipe['depth'],
                        'prev_repairs': pipe['prev_repairs'],
                        'last_inspection': pipe['last_inspection'],
                        'traffic_load': pipe['traffic_load'],
                        'geometry': pipe['geometry']
                    })
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        # Convert to GeoDataFrame for spatial analysis
        if leak_events:
            leak_gdf = gpd.GeoDataFrame(leak_events, geometry='geometry')
            return leak_gdf
        else:
            return gpd.GeoDataFrame()
    
    def generate_simulation_data(self, num_pipes=1000, timespan_days=365, seasonal_factor=True):
        """
        Generate a complete simulation dataset including pipe network and leak events
        
        Parameters:
        -----------
        num_pipes : int
            Number of pipes in the simulated network
        timespan_days : int
            Number of days to simulate
        seasonal_factor : bool
            Whether to include seasonal effects on leak probabilities
            
        Returns:
        --------
        tuple
            (pipe_network_gdf, leak_events_gdf)
        """
        print(f"Generating pipe network with {num_pipes} pipes...")
        pipe_network = self._generate_pipe_network(num_pipes)
        
        print(f"Simulating leaks over {timespan_days} days...")
        leak_events = self._generate_leak_events(pipe_network, timespan_days, seasonal_factor)
        
        print(f"Simulation complete. Generated {len(leak_events)} leak events.")
        return pipe_network, leak_events
    
    def augment_data_with_environmental_factors(self, pipe_gdf, leak_gdf):
        """
        Add environmental factors to the dataset to enrich the simulation
        
        Parameters:
        -----------
        pipe_gdf : GeoDataFrame
            Pipe network data
        leak_gdf : GeoDataFrame
            Leak event data
            
        Returns:
        --------
        tuple
            (augmented_pipe_gdf, augmented_leak_gdf)
        """
        # Add elevation data (simulated)
        pipe_gdf['elevation'] = np.random.uniform(0, 1000, size=len(pipe_gdf))
        
        # Add ground movement risk (simulated earthquake potential)
        pipe_gdf['ground_movement_risk'] = np.random.uniform(0, 10, size=len(pipe_gdf))
        
        # Add proximity to water bodies (simulated)
        pipe_gdf['water_proximity'] = np.random.uniform(0, 5000, size=len(pipe_gdf))
        
        # Add temperature fluctuation level at pipe location
        pipe_gdf['temp_fluctuation'] = np.random.uniform(5, 50, size=len(pipe_gdf))
        
        # Update leak data with these new factors
        if not leak_gdf.empty:
            augmented_leak_gdf = leak_gdf.copy()
            # Match environmental factors from pipe data to leak data
            for env_factor in ['elevation', 'ground_movement_risk', 'water_proximity', 'temp_fluctuation']:
                augmented_leak_gdf[env_factor] = augmented_leak_gdf['pipe_id'].map(
                    pipe_gdf.set_index('pipe_id')[env_factor])
            
            return pipe_gdf, augmented_leak_gdf
        else:
            return pipe_gdf, leak_gdf
    
    def create_feature_dataset_for_ml(self, pipe_gdf, leak_gdf, features_timespan=90):
        """
        Create a feature dataset for machine learning based on historical data
        
        Parameters:
        -----------
        pipe_gdf : GeoDataFrame
            Pipe network data
        leak_gdf : GeoDataFrame
            Leak event data
        features_timespan : int
            Number of days to include in feature computation
            
        Returns:
        --------
        DataFrame
            Features for ML model training
        """
        if leak_gdf.empty:
            print("No leak events to create features from.")
            return pd.DataFrame()
        
        # Make a copy of the pipe data as the base for our features
        features_df = pipe_gdf.copy()
        
        # Convert to dataframe (drop geometry for ML)
        features_df = pd.DataFrame(features_df.drop(columns=['geometry']))
        
        # Add leak history features
        if not leak_gdf.empty:
            # Sort leaks by date
            leak_gdf = leak_gdf.sort_values('date')
            
            # Calculate leak frequency by pipe
            leak_counts = leak_gdf.groupby('pipe_id').size().reset_index(name='leak_count')
            features_df = features_df.merge(leak_counts, on='pipe_id', how='left')
            features_df['leak_count'] = features_df['leak_count'].fillna(0)
            
            # Calculate days since last leak
            last_leak_date = leak_gdf.groupby('pipe_id')['date'].max().reset_index()
            last_leak_date = last_leak_date.rename(columns={'date': 'last_leak_date'})
            
            features_df = features_df.merge(last_leak_date, on='pipe_id', how='left')
            latest_date = leak_gdf['date'].max()
            
            features_df['days_since_last_leak'] = (latest_date - features_df['last_leak_date']).dt.days
            features_df['days_since_last_leak'] = features_df['days_since_last_leak'].fillna(365*5)  # No leak = 5 years
            features_df = features_df.drop(columns=['last_leak_date'])
            
            # Calculate average repair cost
            avg_cost = leak_gdf.groupby('pipe_id')['repair_cost'].mean().reset_index()
            avg_cost = avg_cost.rename(columns={'repair_cost': 'avg_repair_cost'})
            features_df = features_df.merge(avg_cost, on='pipe_id', how='left')
            features_df['avg_repair_cost'] = features_df['avg_repair_cost'].fillna(0)
            
            # Add target variable: had_leak_recently (last X days)
            recent_date = latest_date - datetime.timedelta(days=features_timespan)
            recent_leaks = leak_gdf[leak_gdf['date'] >= recent_date]
            recent_leak_pipes = recent_leaks['pipe_id'].unique()
            
            features_df['had_leak_recently'] = features_df['pipe_id'].isin(recent_leak_pipes).astype(int)
        
        # Convert categorical variables to one-hot encoding
        features_df = pd.get_dummies(features_df, columns=['material', 'diameter', 'soil_type'])
        
        # Drop identification columns not useful for prediction
        features_df = features_df.drop(columns=['pipe_id'])
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df

# Example usage
if __name__ == "__main__":
    simulator = PipeLeakSimulator(seed=42)
    pipe_network, leak_events = simulator.generate_simulation_data(num_pipes=1000, timespan_days=365)
    
    # Add environmental factors
    pipe_network, leak_events = simulator.augment_data_with_environmental_factors(pipe_network, leak_events)
    
    # Create feature dataset for ML
    features_df = simulator.create_feature_dataset_for_ml(pipe_network, leak_events)
    
    # Display sample data
    print("\nPipe Network Sample:")
    print(pipe_network.head())
    
    print("\nLeak Events Sample:")
    if not leak_events.empty:
        print(leak_events.head())
    else:
        print("No leak events generated.")
    
    print("\nML Features Sample:")
    print(features_df.head())
    
    # Save data to CSV files
    pipe_network.drop(columns=['geometry']).to_csv("pipe_network.csv", index=False)
    if not leak_events.empty:
        leak_events.drop(columns=['geometry']).to_csv("leak_events.csv", index=False)
    features_df.to_csv("ml_features.csv", index=False)
    
    print("\nSimulation data saved to CSV files.")