import os
import sys
import glob
from pathlib import Path

# --- 1. АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПУТЕЙ ---
# Получаем абсолютный путь к текущей директории скрипта и корню проекта
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# --- 2. НАСТРОЙКА HADOOP (ДЛЯ WINDOWS) ---
# Указываем Spark, что winutils.exe находится в папке hadoop/bin внутри проекта
HADOOP_HOME = os.path.join(PROJECT_ROOT, "hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] += os.pathsep + os.path.join(HADOOP_HOME, "bin")

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, Window, functions as F


def run_task3_spark():
    print("Starting Spark Session...")
    spark = (SparkSession.builder.master("local[*]")
             .appName("task3_anomaly_detection")
             .config("spark.driver.memory", "4g").getOrCreate())

    # Отключаем лишний спам в консоли (WARN сообщения)
    spark.sparkContext.setLogLevel("ERROR")

    # --- 3. ПУТЬ К ДАННЫМ ---
    pattern = os.path.join(PROJECT_ROOT, "data/raw_telemetry", "*.csv")
    files = [Path(p).resolve().as_uri() for p in glob.glob(pattern)]

    print(f"Loading {len(files)} files...")

    # Защита от ошибки, если файлы всё-таки не найдены
    if len(files) == 0:
        print(f"❌ ОШИБКА: Файлы не найдены по пути: {pattern}")
        print("Убедитесь, что в папке samples есть csv-файлы!")
        return

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
    # Сохраняем в текущую папку (data_processor/3/)
    output_path = os.path.join(CURRENT_DIR, "task3_full_season_anomalies.csv")
    anomalies.toPandas().to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")


if __name__ == "__main__":
    run_task3_spark()