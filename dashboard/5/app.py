import streamlit as st
import pandas as pd
import os
from PIL import Image

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="F1 Analytics Task 5",
    page_icon="🏎️",
    layout="wide"
)

# --- ПУТИ К ФАЙЛАМ ---
# Так как мы в dashboard/5/app.py, то корень проекта на 2 уровня выше
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data_processor", "5", "reports")
CSV_COMPARISON = os.path.join(REPORTS_DIR, "predictions_comparison.csv")
AGG_FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "task5_aggregated_features.csv")

# --- ЗАГОЛОВОК ---
st.title("🏎️ Задача 5: Прогноз результатов (Apache Spark)")
st.markdown(
    "Построение модели машинного обучения на основе телеметрии с использованием `PySpark DataFrame` и `Spark MLlib`.")

# --- ВКЛАДКИ ДАШБОРДА ---
tab1, tab2 = st.tabs(["🎯 Прогноз и Аналитика", "📊 Исходные данные (Features)"])

with tab1:
    st.header("Оценка точности прогноза (Spark MLlib)")
    st.markdown("Модель линейной регрессии обучена на телеметрии первых 15 кругов (темп, скорость, шины, погода).")

    # 1. Загрузка и показ графиков (Используем width="stretch" вместо устаревших параметров)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Точность прогноза")
        img_actual = os.path.join(REPORTS_DIR, "actual_vs_predicted.png")
        if os.path.exists(img_actual):
            image1 = Image.open(img_actual)
            st.image(image1, width="stretch")
        else:
            st.warning("График точности не найден. Сначала запустите скрипт data_processor/5/task5_spark.py")

    with col2:
        st.subheader("Влияние факторов")
        img_features = os.path.join(REPORTS_DIR, "feature_importance.png")
        if os.path.exists(img_features):
            image2 = Image.open(img_features)
            st.image(image2, width="stretch")
        else:
            st.warning("График важности факторов не найден.")

    st.divider()

    # 2. Интерактивная таблица предсказаний
    st.subheader("Детальное сравнение прогноза с реальными результатами")
    if os.path.exists(CSV_COMPARISON):
        df_comp = pd.read_csv(CSV_COMPARISON)

        # Интерактивный фильтр по Гран-при
        gps = ["Все этапы"] + list(df_comp['GrandPrix'].unique())
        selected_gp = st.selectbox("Выберите Гран-при для фильтрации:", gps)

        if selected_gp != "Все этапы":
            df_display = df_comp[df_comp['GrandPrix'] == selected_gp]
        else:
            df_display = df_comp


        # Подсветка ошибок
        def highlight_errors(val):
            if pd.isna(val): return ''
            if val <= 2:
                return 'background-color: #2e7d32; color: white'
            elif val <= 5:
                return 'background-color: #f57f17; color: white'
            else:
                return 'background-color: #c62828; color: white'


        st.dataframe(
            df_display.style.map(highlight_errors, subset=['Error (Delta)']),
            width="stretch",
            height=400
        )
    else:
        st.error("Файл с предсказаниями не найден.")

with tab2:
    st.header("Сформированная база признаков (Aggregated Dataset)")
    st.markdown(
        "Сводные данные, сагрегированные с помощью **PySpark** из 4+ ГБ сырой телеметрии `data/raw_telemetry/`.")

    if os.path.exists(AGG_FEATURES_CSV):
        df_features = pd.read_csv(AGG_FEATURES_CSV)
        st.dataframe(df_features, width="stretch")
        st.markdown(f"**Всего записей:** {df_features.shape[0]} | **Признаков:** {df_features.shape[1]}")
    else:
        st.info("Агрегированный датасет пока не сформирован.")