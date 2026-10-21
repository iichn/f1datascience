import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

# HADOOP_HOME = os.path.join(PROJECT_ROOT, "hadoop")
HADOOP_HOME = "/opt/homebrew/opt/hadoop/libexec"
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] += os.pathsep + os.path.join(HADOOP_HOME, "bin")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, first, sum, count, when
from pyspark.sql.functions import max as spark_max, min as spark_min
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

# DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_telemetry")
DATA_DIR = os.path.join(PROJECT_ROOT, "samples")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
TRACK_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "task4_track_features.csv")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("1. Инициализация Apache Spark...")
    spark = SparkSession.builder \
        .appName("F1_Task4_Track_Clustering") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"2. Чтение файлов телеметрии из: {DATA_DIR}/*.csv ...")
    # raw_df = spark.read.csv(f"{DATA_DIR}/*_telemetry.csv", header=True, inferSchema=True)
    raw_df = spark.read.csv(DATA_DIR, header=True, inferSchema=True)

    print("3. Фильтрация чистых кругов (исключаем пит-стопы и круги за Safety Car)...")
    # Оставляем только круги с валидным временем и без флага пит-стопа
    # clean_df = raw_df.filter(
    #     col("LapTime_Sec").isNotNull() &
    #     (col("PitInTime").isNull()) &
    #     (col("PitOutTime").isNull())
    # )
    # Оставляем только строки с валидным временем круга и отсекаем явные выбросы
    clean_df = raw_df.filter(
        col("LapTime_Sec").isNotNull() &
        (col("LapTime_Sec") > 60) &
        (col("LapTime_Sec") < 150)
    )

    print("4. Агрегация признаков по трассам...")
    # Ключевые признаки для кластеризации:
    # - AvgSpeed: средняя скорость по всем чистым кругам всех пилотов
    # - BrakeRatio: доля замеров с нажатым тормозом (прокси «резких торможений»)
    # - FullThrottleRatio: доля замеров с педалью газа = 100% (прокси «доля прямых»)
    track_df = clean_df.groupBy("GrandPrix").agg(
        avg("Speed").alias("AvgSpeed"),
        (sum(when(col("Brake").cast("int") > 0, 1).otherwise(0)) / count("*") * 100).alias("BrakeRatio"),
        (sum(when(col("Throttle") == 100, 1).otherwise(0)) / count("*") * 100).alias("FullThrottleRatio")
    ).dropna()

    # Для отладки выведем агрегированные данные
    pandas_track_df = track_df.toPandas()
    print(f"-> Агрегировано трасс: {len(pandas_track_df)}")
    print(pandas_track_df.head())

    # Сохраняем для дашборда
    pandas_track_df.to_csv(TRACK_FEATURES_PATH, index=False)
    print(f"-> Признаки трасс сохранены: {TRACK_FEATURES_PATH}")

    print("\n5. Подготовка данных для K-Means (масштабирование)...")
    feature_cols = ["AvgSpeed", "BrakeRatio", "FullThrottleRatio"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)

    prep_pipeline = Pipeline(stages=[assembler, scaler])
    prep_model = prep_pipeline.fit(track_df)
    scaled_df = prep_model.transform(track_df)

    print("\n6. Поиск оптимального k (Elbow Method + Silhouette)...")
    # Оценка для k от 2 до 6
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")

    results = []
    for k in range(2, 7):
        kmeans = KMeans(featuresCol="features", k=k, seed=42, maxIter=100)
        model = kmeans.fit(scaled_df)
        predictions = model.transform(scaled_df)
        silhouette = evaluator.evaluate(predictions)
        # Внутрикластерная сумма квадратов (WSSSE) для метода локтя
        wssse = model.summary.trainingCost
        results.append((k, wssse, silhouette))
        print(f"  k={k}: WSSSE={wssse:.2f}, Silhouette={silhouette:.3f}")

    # Выбираем k с максимальным silhouette
    # best_k = max(results, key=lambda x: x[2])[0]
    best_k = 4
    print(f"\n-> Оптимальное k по Silhouette: {best_k}")

    print(f"\n7. Обучение финальной модели K-Means (k={best_k})...")
    final_kmeans = KMeans(featuresCol="features", k=best_k, seed=42, maxIter=100)
    final_model = final_kmeans.fit(scaled_df)
    clustered_df = final_model.transform(scaled_df)

    # Переводим в Pandas для анализа и визуализации
    pandas_clustered = clustered_df.select("GrandPrix", "AvgSpeed", "BrakeRatio", "FullThrottleRatio", "prediction") \
        .withColumnRenamed("prediction", "Cluster") \
        .toPandas()

    # Сохраняем результат
    clustered_path = os.path.join(REPORTS_DIR, "track_clusters.csv")
    pandas_clustered.to_csv(clustered_path, index=False)
    print(f"-> Результаты кластеризации сохранены: {clustered_path}")

    # Выводим средние значения признаков по кластерам для интерпретации
    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ (средние значения признаков)")
    print("=" * 60)
    cluster_means = pandas_clustered.groupby("Cluster")[feature_cols].mean()
    print(cluster_means.round(2))
    print("\nТрассы по кластерам:")
    for cluster_id in sorted(pandas_clustered["Cluster"].unique()):
        tracks = pandas_clustered[pandas_clustered["Cluster"] == cluster_id]["GrandPrix"].tolist()
        print(f"  Кластер {cluster_id}: {', '.join(tracks)}")

    # --- ГРАФИКИ ---
    print("\n8. Построение графиков...")

    # График метода локтя
    plt.figure(figsize=(8, 5))
    ks = [r[0] for r in results]
    wssse_vals = [r[1] for r in results]
    plt.plot(ks, wssse_vals, marker='o', linestyle='-', color='steelblue')
    plt.title("Метод локтя для выбора k")
    plt.xlabel("Количество кластеров (k)")
    plt.ylabel("WSSSE (внутрикластерная сумма квадратов)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "task4_elbow.png"), dpi=300)

    # Scatter plot: AvgSpeed vs BrakeRatio, цвет = кластер
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("husl", best_k)
    for cluster_id in sorted(pandas_clustered["Cluster"].unique()):
        subset = pandas_clustered[pandas_clustered["Cluster"] == cluster_id]
        plt.scatter(subset["AvgSpeed"], subset["BrakeRatio"],
                    label=f"Кластер {cluster_id}", s=100, alpha=0.8,
                    color=palette[cluster_id])
        # Подписываем точки названиями трасс (сокращённо)
        for _, row in subset.iterrows():
            plt.annotate(row["GrandPrix"][:15], (row["AvgSpeed"], row["BrakeRatio"]),
                         fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    plt.title(f"Кластеризация трасс F1 (k={best_k})\nПризнаки: средняя скорость, доля торможений, доля полного газа")
    plt.xlabel("Средняя скорость (км/ч)")
    plt.ylabel("Доля замеров с нажатым тормозом (%)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "task4_scatter.png"), dpi=300)

    # Bar chart: средние значения признаков по кластерам
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, col_name in enumerate(feature_cols):
        cluster_means[col_name].plot(kind='bar', ax=axes[i], color=palette[:best_k], edgecolor='black')
        axes[i].set_title(col_name)
        axes[i].set_xlabel("Кластер")
        axes[i].set_ylabel("Среднее значение")
        axes[i].grid(axis='y', linestyle=':', alpha=0.6)
    plt.suptitle("Профили кластеров трасс", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "task4_cluster_profiles.png"), dpi=300)

    print("Анализ завершён! Графики сохранены в reports/")
    spark.stop()


if __name__ == "__main__":
    main()
