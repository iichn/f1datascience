import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

HADOOP_HOME = os.path.join(PROJECT_ROOT, "hadoop")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.environ["hadoop.home.dir"] = HADOOP_HOME
os.environ["PATH"] += os.pathsep + os.path.join(HADOOP_HOME, "bin")

# Импорты PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, max, min, first, sum, count, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# --- ПУТИ ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_telemetry")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
AGG_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "task5_aggregated_features.csv")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("1. Инициализация Apache Spark...")
    # Создаем локальную сессию Spark, выделяем ей память
    spark = SparkSession.builder \
        .appName("F1_Task5_Spark_ML") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Скрываем лишние логи Spark в консоли
    spark.sparkContext.setLogLevel("ERROR")

    print(f"2. Параллельное чтение файлов телеметрии из: {DATA_DIR}/*.csv ...")
    # Spark читает ВСЕ 44 файла (4+ ГБ) параллельно одной командой!
    raw_df = spark.read.csv(f"{DATA_DIR}/*_telemetry.csv", header=True, inferSchema=True)

    print("3. Распределенная обработка данных (MapReduce под капотом)...")
    # Оставляем только первые 15 кругов
    early_df = raw_df.filter(col("LapNumber") <= 15)

    # Агрегация (GroupBy) по Гран-при и Пилоту
    agg_df = early_df.groupBy("GrandPrix", "Driver").agg(
        first("Team").alias("Team"),
        first("GridPosition").cast("double").alias("GridPosition"),
        first("FinalPosition").cast("double").alias("FinalPosition"),
        first("Compound").alias("Compound"),
        avg("Speed").alias("AvgEarlySpeed"),
        max("Speed").alias("MaxEarlySpeed"),
        avg("LapTime_Sec").alias("AvgEarlyLapTime"),
        min("LapTime_Sec").alias("BestEarlyLapTime"),
        (sum(when(col("Throttle") == 100, 1).otherwise(0)) / count("*") * 100).alias("PctFullThrottle"),
        (sum(when(col("Brake").cast("int") > 0, 1).otherwise(0)) / count("*") * 100).alias("PctBraking"),
        first("AirTemp").cast("double").alias("AirTemp"),
        first("TrackTemp").cast("double").alias("TrackTemp"),
        first("Rainfall").cast("double").alias("Rainfall")
    ).dropna(subset=["FinalPosition", "GridPosition"])

    # Запускаем вычисления и выгружаем сагрегированные данные в Pandas
    # (после группировки это всего ~200 строк, поэтому toPandas() работает мгновенно)
    pandas_agg_df = agg_df.toPandas()

    # СОХРАНЯЕМ ФАЙЛ ДЛЯ ДАШБОРДА (Исправление Вашего 3-го пункта)
    pandas_agg_df.to_csv(AGG_FILE_PATH, index=False)
    print(f"-> Агрегированный датасет сохранен для дашборда: {AGG_FILE_PATH}")

    print("\n4. Построение конвейера машинного обучения (Spark MLlib)...")
    # Кодируем строковые колонки в числа
    team_indexer = StringIndexer(inputCol="Team", outputCol="Team_Idx", handleInvalid="keep")
    comp_indexer = StringIndexer(inputCol="Compound", outputCol="Compound_Idx", handleInvalid="keep")

    # Заполняем пропуски во временах круга
    imputer = Imputer(inputCols=["AvgEarlyLapTime", "BestEarlyLapTime"],
                      outputCols=["AvgEarlyLapTime_imp", "BestEarlyLapTime_imp"]).setStrategy("median")

    # Собираем все признаки в единый вектор для модели
    feature_cols = [
        "GridPosition", "Team_Idx", "Compound_Idx", "AvgEarlySpeed",
        "MaxEarlySpeed", "AvgEarlyLapTime_imp", "BestEarlyLapTime_imp",
        "PctFullThrottle", "PctBraking", "AirTemp", "TrackTemp", "Rainfall"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    # Используем Линейную Регрессию (аналог Ridge из Scikit-Learn)
    lr = LinearRegression(featuresCol="features", labelCol="FinalPosition", regParam=0.1)

    # Собираем пайплайн (Конвейер)
    pipeline = Pipeline(stages=[team_indexer, comp_indexer, imputer, assembler, lr])

    # Разбиваем на Train / Test
    train_data, test_data = agg_df.randomSplit([0.8, 0.2], seed=42)

    print("5. Обучение модели...")
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ SPARK MLLIB (LINEAR REGRESSION)")
    print("=" * 60)

    # Оценка качества модели
    evaluator_mae = RegressionEvaluator(labelCol="FinalPosition", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="FinalPosition", predictionCol="prediction", metricName="r2")

    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print(f"MAE  (Средняя ошибка в позициях): ±{mae:.2f}")
    print(f"R^2  (Коэффициент детерминации):   {r2:.3f}")

    # Переводим предсказания в Pandas для отрисовки графиков и сохранения отчета
    df_preds = predictions.select("GrandPrix", "Driver", "GridPosition", "FinalPosition", "prediction").toPandas()
    df_preds['PredictedRank'] = np.clip(np.round(df_preds['prediction']), 1, 20).astype(int)
    df_preds['Error (Delta)'] = np.abs(df_preds['FinalPosition'] - df_preds['PredictedRank']).astype(int)

    # Сохраняем сравнительную таблицу
    csv_comp_path = os.path.join(REPORTS_DIR, "predictions_comparison.csv")
    df_preds.to_csv(csv_comp_path, index=False)

    # --- ГРАФИКИ ---
    print("\n6. Построение графиков...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=df_preds['FinalPosition'], y=df_preds['PredictedRank'], color='crimson', alpha=0.7, s=70)
    plt.plot([1, 20], [1, 20], color='navy', linestyle='--', label='Идеальный прогноз (y = x)')
    plt.title(f"Точность прогноза (Spark MLlib)\nСредняя ошибка (MAE) = ±{mae:.2f} позиции", fontsize=12)
    plt.xlabel("Реальная позиция на финише")
    plt.ylabel("Предсказанная позиция")
    plt.xlim(0.5, 20.5);
    plt.ylim(0.5, 20.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "actual_vs_predicted.png"), dpi=300)

    # Достаем веса из модели линейной регрессии (последний этап в пайплайне)
    lr_model = model.stages[-1]
    importances = np.abs(lr_model.coefficients.toArray())

    plt.figure(figsize=(10, 6))
    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    imp_series.plot(kind='barh', color='teal', edgecolor='black')
    plt.title("Влияние факторов на итоговый результат гонки (Spark MLlib)", fontsize=12)
    plt.xlabel("Абсолютный вес коэффициента")
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=300)

    print("Анализ завершен! Можно открывать дашборд.")
    spark.stop()


if __name__ == "__main__":
    main()