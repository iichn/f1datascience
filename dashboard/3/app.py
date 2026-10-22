import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

st.set_page_config(page_title="Task 3: F1 Anomaly Dashboard", layout="wide")
st.title("🏎️ Task 3: F1 Anomaly Detection (Spark + Streamlit)")
st.write("This dashboard visualizes anomalies processed by Apache Spark, distinguishing between normal lock-ups and technical failures.")

# --- PATH SETUP ---
# Определяем пути относительно текущей папки (dashboard/3)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Выходим на два уровня вверх к корню проекта (f1datascience)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Файл с аномалиями лежит там, куда его сохранил Spark
anomaly_file = os.path.join(project_root, 'data_processor', '3', 'task3_full_season_anomalies.csv')
# Сырые телеметрии лежат в папке samples
samples_dir = os.path.join(project_root, 'samples')

# The full 2022 driver grid
ALL_DRIVERS = ["VER", "PER", "LEC", "SAI", "HAM", "RUS", "NOR", "RIC", "ALO", "OCO", "GAS", "TSU", "VET", "STR", "MAG", "MSC", "BOT", "ZHO", "ALB", "LAT"]

if not os.path.exists(anomaly_file):
    st.error(f"Could not find {anomaly_file}. Make sure the Spark CSV is in the folder!")
else:
    # Load the Spark anomalies
    df = pd.read_csv(anomaly_file)
    
    st.sidebar.header("Filters")
    
    # Dropdowns
    gp = st.sidebar.selectbox("Select Grand Prix:", sorted(df['GrandPrix'].unique()))
    session = st.sidebar.selectbox("Select Session:", ["Q", "R"])
    driver = st.sidebar.selectbox("Select Driver:", ALL_DRIVERS)
    
    st.write(f"### 🚨 Anomalies for {driver} at {gp} ({session})")
    
    # Filter the data
    driver_anomalies = df[(df['GrandPrix'] == gp) & (df['Session'] == session) & (df['Driver'] == driver)]
    
    if len(driver_anomalies) == 0:
        st.success(f"✅ Great news! No anomalies detected for {driver} in this session.")
    else:
        st.write(f"Found {len(driver_anomalies)} anomalies.")
        
        # Show the table with the NEW Anomaly_Reason column
        st.dataframe(driver_anomalies[['Anomaly_Reason', 'LapNumber', 'Distance', 'Speed', 'Speed_Diff', 'RPM', 'RPM_Diff']])
        # --- PLOTTING THE FULL LAP ---
        # Find the original raw sample file to draw the background line
        gp_name_formatted = gp.replace(" ", "_")
        search_pattern = os.path.join(samples_dir, f"*{gp_name_formatted}*{session}_telemetry.csv")
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            full_df = pd.read_csv(matching_files[0])
            target_lap = driver_anomalies['LapNumber'].iloc[0]
            
            lap_telemetry = full_df[(full_df['Driver'] == driver) & (full_df['LapNumber'] == target_lap)].copy()
            lap_telemetry = lap_telemetry.sort_values(by='Distance')
            
            if len(lap_telemetry) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Top Graph: Speed
                ax1.plot(lap_telemetry['Distance'], lap_telemetry['Speed'], color='blue', alpha=0.5, label='Full Lap Telemetry')
                
                # Color-code the anomalies based on the Reason
                for index, row in driver_anomalies.iterrows():
                    if row['LapNumber'] == target_lap and row['Speed'] > 0:
                        dot_color = 'red' if "Phantom" in str(row['Anomaly_Reason']) else 'orange'
                        ax1.scatter(row['Distance'], row['Speed'], color=dot_color, s=100, zorder=5)
                
                ax1.set_ylabel('Speed (km/h)')
                ax1.set_title(f'{driver} - {gp} ({session}) - Lap {target_lap}')
                
                # Custom Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Phantom Braking (Tech)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Heavy Braking (Lock-up)')
                ]
                ax1.legend(handles=legend_elements, loc='upper right')
                
                # Bottom Graph: RPM
                ax2.plot(lap_telemetry['Distance'], lap_telemetry['RPM'], color='gray', alpha=0.5, label='Full Lap Telemetry')
                plot_rpm = driver_anomalies[(driver_anomalies['RPM'] > 0) & (driver_anomalies['LapNumber'] == target_lap)]
                if len(plot_rpm) > 0:
                    ax2.scatter(plot_rpm['Distance'], plot_rpm['RPM'], color='purple', s=100, label='RPM Anomaly')
                    
                ax2.set_ylabel('RPM')
                ax2.set_xlabel('Distance (m)')
                ax2.legend()
                
                st.pyplot(fig)