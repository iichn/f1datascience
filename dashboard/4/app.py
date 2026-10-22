import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Task 4: Track Clustering", layout="wide")
st.title("🏁 Task 4: F1 Track Clustering")
st.write("Результаты кластеризации гоночных трасс с помощью алгоритма K-Means (Apache Spark).")

# --- НАСТРОЙКА ПУТЕЙ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Указываем путь к папке reports, которую создал твой Spark-скрипт
reports_dir = os.path.join(project_root, 'data_processor', '4', 'reports')
clusters_file = os.path.join(reports_dir, 'track_clusters.csv')

if not os.path.exists(clusters_file):
    st.error(f"❌ Файл не найден: {clusters_file}")
    st.warning("Убедитесь, что вы запускали скрипт из папки data_processor/4")
else:
    # 1. Читаем данные
    df = pd.read_csv(clusters_file)

    # 2. Показываем распределение трасс по кластерам
    st.subheader("🏎️ Состав кластеров")
    clusters = sorted(df['Cluster'].unique())

    for c in clusters:
        tracks = df[df['Cluster'] == c]['GrandPrix'].tolist()
        st.info(f"**Кластер {c}** ({len(tracks)} трасс): {', '.join(tracks)}")

    # 3. Выводим таблицу (ИСПРАВЛЕНО ДЛЯ НОВЫХ ВЕРСИЙ STREAMLIT)
    st.subheader("📊 Таблица данных")
    st.dataframe(df, width='stretch')

    # 4. Показываем графики, которые УЖЕ сохранил Spark
    st.subheader("📈 Графики кластеризации")

    scatter_img = os.path.join(reports_dir, "task4_scatter.png")
    profiles_img = os.path.join(reports_dir, "task4_cluster_profiles.png")
    elbow_img = os.path.join(reports_dir, "task4_elbow.png")

    # Выводим картинки, если они существуют
    if os.path.exists(scatter_img):
        st.image(scatter_img, caption="Проекция трасс (Скорость vs Торможение)")

    if os.path.exists(profiles_img):
        st.image(profiles_img, caption="Профили признаков по кластерам")

    if os.path.exists(elbow_img):
        with st.expander("Посмотреть график метода локтя"):
            st.image(elbow_img)