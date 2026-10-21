import os
import sys
import glob
from pathlib import Path


os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, Window, functions as F

def run_task3_spark():
    print("Starting Spark Session...")
    spark = (SparkSession.builder.master("local[*]").appName("task3_anomaly_detection")
             .config("spark.driver.memory", "4g").getOrCreate())

    # 1. Load the samples from the local folder
    # When moving to Big Data, change this to: "../../data/raw_telemetry/*.csv"
    pattern = "../../samples/*.csv" 
    files = [Path(p).resolve().as_uri() for p in glob.glob(pattern)]
    
    print(f"Loading {len(files)} files...")
    df = spark.read.csv(files, header=True, inferSchema=True)

    # 2. The Math (Window Functions)
    windowSpec = Window.partitionBy("Year", "GrandPrix", "Session", "Driver", "LapNumber").orderBy("Distance")

    df = df.withColumn("Prev_Speed", F.lag("Speed").over(windowSpec))
    df = df.withColumn("Speed_Diff", F.col("Speed") - F.col("Prev_Speed"))

    df = df.withColumn("Prev_RPM", F.lag("RPM").over(windowSpec))
    df = df.withColumn("RPM_Diff", F.col("RPM") - F.col("Prev_RPM"))

    # 3. Find Anomalies (Threshold)
    anomalies = df.filter(
        (F.col("TrackStatus").cast("int") == 1) & 
        (F.col("Speed") > 0) &
        ((F.col("Speed_Diff") < -50) | (F.col("RPM_Diff") < -4000))
    )

    # 4. THE HYBRID UPGRADE: Contextual Classification
    anomalies = anomalies.withColumn(
        "Anomaly_Reason",
        F.when(
            (F.col("Speed_Diff") < -50) & 
            ((F.col("Brake").isNull()) | (F.col("Brake") == False)), 
            "Phantom Braking (Tech Failure)"
        ).otherwise("Heavy Braking / Lock-up")
    )

    print(f"✅ Found {anomalies.count()} total anomalies!")
    
    # 5. Save to CSV (This will overwrite the old CSV in the folder)
    output_path = "task3_full_season_anomalies.csv"
    anomalies.toPandas().to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")

if __name__ == "__main__":
    run_task3_spark()