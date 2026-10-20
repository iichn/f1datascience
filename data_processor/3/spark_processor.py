from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col, lag
import os

def run_task3_spark():
    # 1. Start Spark
    spark = SparkSession.builder.appName("F1_Task3_Anomaly_Detection").getOrCreate()

    # 2. Define the path
    # Right now, point to samples. When Big Data is ready, change to 'data/raw_telemetry/*.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Use this for testing:
    file_path = os.path.join(current_dir, '..', '..', 'samples', '*.csv')
    # Use this for the Big Data phase (uncomment later):
    # file_path = os.path.join(current_dir, '..', '..', 'data', 'raw_telemetry', '*.csv')

    print(f"Loading data from: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    print(f"✅ Total rows loaded: {df.count()}")

    # 3. Window Functions (The Math)
    windowSpec = Window.partitionBy("Year", "GrandPrix", "Session", "Driver", "LapNumber").orderBy("Distance")

    df = df.withColumn("Prev_Speed", lag("Speed").over(windowSpec))
    df = df.withColumn("Speed_Diff", col("Speed") - col("Prev_Speed"))

    df = df.withColumn("Prev_RPM", lag("RPM").over(windowSpec))
    df = df.withColumn("RPM_Diff", col("RPM") - col("Prev_RPM"))

    # 4. Filter for Anomalies
    # Speed drop > 50 km/h OR RPM drop > 4000, ONLY on Green Flag (TrackStatus == 1)
    anomalies = df.filter(
        (col("TrackStatus") == 1) & 
        ((col("Speed_Diff") < -50) | (col("RPM_Diff") < -4000))
    )

    print(f"🚨 Found {anomalies.count()} anomalies!")

    # 5. Save results for Streamlit
    output_path = os.path.join(current_dir, 'task3_full_season_anomalies.csv')
    anomalies.toPandas().to_csv(output_path, index=False)
    print(f"✅ Saved anomalies to {output_path}")

if __name__ == "__main__":
    run_task3_spark()