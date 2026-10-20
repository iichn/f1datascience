import os
import sys

import glob
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, Window, functions as F


def build_laps_spark(pattern="data/raw_telemetry/*.csv"):
    spark = (SparkSession.builder.master("local[*]").appName("task1_tyre_wear")
             .config("spark.driver.memory", "4g").getOrCreate())
    files = [Path(p).resolve().as_uri() for p in glob.glob(pattern)]
    raw = spark.read.csv(files, header=True, inferSchema=True)

    green = F.coalesce(F.col("TrackStatus").cast("int") == 1, F.lit(False)).cast("int")
    laps = raw.groupBy("Year", "GrandPrix", "Driver", "LapNumber").agg(
        F.first("Team").alias("Team"),
        F.first("Compound").alias("Compound"),
        F.first("TyreLife").alias("TyreLife"),
        F.first("Stint").alias("Stint"),
        F.first("LapTime_Sec").alias("LapTime"),
        F.avg("Speed").alias("AvgSpeed"),
        F.max("Speed").alias("MaxSpeed"),
        F.avg((F.col("Throttle") >= 99).cast("double")).alias("FullThrottle"),
        F.first("TrackTemp").alias("TrackTemp"),
        (F.min(green) == 1).alias("Green"),
    )

    w = Window.partitionBy("Year", "GrandPrix", "Driver", "Stint")
    laps = laps.withColumn(
        "InOut",
        (F.col("LapNumber") == F.min("LapNumber").over(w)) | (F.col("LapNumber") == F.max("LapNumber").over(w)),
    )
    return laps.toPandas()