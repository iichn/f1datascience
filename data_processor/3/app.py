import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

st.set_page_config(page_title="Task 3: F1 Anomaly Dashboard", layout="wide")

st.title("🏎️ Task 3: 2022 F1 Anomaly Detection (Spark + Streamlit)")

current_dir = os.path.dirname(os.path.abspath(__file__))
anomaly_file = os.path.join(current_dir, 'task3_full_season_anomalies.csv')
samples_dir = os.path.join(current_dir, '..', '..', 'samples')

# The full 2022 driver grid
ALL_DRIVERS = ["VER", "PER", "LEC", "SAI", "HAM", "RUS", "NOR", "RIC", "ALO", "OCO", "GAS", "TSU", "VET", "STR", "MAG", "MSC", "BOT", "ZHO", "ALB", "LAT"]

if not os.path.exists(anomaly_file):
    st.error(f"Could not find {anomaly_file}. Make sure you copied the Spark output!")
else:
    # 1. Load the Spark anomalies
    df = pd.read_csv(anomaly_file)
    
    st.sidebar.header("Filters")
    
    # 2. Create the dropdowns
    gp = st.sidebar.selectbox("Select Grand Prix:", sorted(df['GrandPrix'].unique()))
    session = st.sidebar.selectbox("Select Session:", ["Q", "R"])
    driver = st.sidebar.selectbox("Select Driver:", ALL_DRIVERS)
    
    st.write(f"### 🚨 Anomalies for {driver} at {gp} ({session})")
    
    # Filter for the specific driver and session
    driver_anomalies = df[(df['GrandPrix'] == gp) & (df['Session'] == session) & (df['Driver'] == driver)]
    
    if len(driver_anomalies) == 0:
        st.success(f"✅ Great news! No anomalies detected for {driver} in this session.")
    else:
        st.write(f"Found {len(driver_anomalies)} anomalies.")
        st.dataframe(driver_anomalies[['LapNumber', 'Distance', 'Speed', 'Speed_Diff', 'RPM', 'RPM_Diff', 'nGear']])
        
        # --- GET THE FULL LAP FOR THE GRAPH ---
        # 3. Find the matching sample file based on the GP name
        # We use a wildcard match to find the file
        gp_name_formatted = gp.replace(" ", "_")
        search_pattern = os.path.join(samples_dir, f"*{gp_name_formatted}*{session}_telemetry.csv")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            st.warning(f"Could not find the sample file to draw the full lap for {gp}.")
        else:
            # Load the full sample file
            full_df = pd.read_csv(matching_files[0])
            
            # Get the lap number of the first anomaly to draw the background line
            target_lap = driver_anomalies['LapNumber'].iloc[0]
            
            # Filter the full data to just this driver and lap
            lap_telemetry = full_df[(full_df['Driver'] == driver) & (full_df['LapNumber'] == target_lap)].copy()
            lap_telemetry = lap_telemetry.sort_values(by='Distance')
            
            if len(lap_telemetry) == 0:
                st.warning(f"Could not find telemetry data for {driver} on Lap {target_lap} in the sample file.")
            else:
                # --- PLOTTING ---
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Top Graph: Speed
                ax1.plot(lap_telemetry['Distance'], lap_telemetry['Speed'], color='blue', alpha=0.5, label='Full Lap Telemetry')
                
                # Overlay Speed anomalies (filter out 0 speed for cleaner graph)
                plot_speed = driver_anomalies[(driver_anomalies['Speed'] > 0) & (driver_anomalies['LapNumber'] == target_lap)]
                if len(plot_speed) > 0:
                    ax1.scatter(plot_speed['Distance'], plot_speed['Speed'], color='red', s=100, label='Speed Anomaly')
                
                ax1.set_ylabel('Speed (km/h)')
                ax1.set_title(f'{driver} - {gp} ({session}) - Lap {target_lap}')
                ax1.legend()
                
                # Bottom Graph: RPM
                ax2.plot(lap_telemetry['Distance'], lap_telemetry['RPM'], color='orange', alpha=0.5, label='Full Lap Telemetry')
                
                # Overlay RPM anomalies
                plot_rpm = driver_anomalies[(driver_anomalies['RPM'] > 0) & (driver_anomalies['LapNumber'] == target_lap)]
                if len(plot_rpm) > 0:
                    ax2.scatter(plot_rpm['Distance'], plot_rpm['RPM'], color='purple', s=100, label='RPM Anomaly')
                    
                ax2.set_ylabel('RPM')
                ax2.set_xlabel('Distance (m)')
                ax2.legend()
                
                st.pyplot(fig)