import streamlit as st
import pandas as pd
import os
import importlib.util
from PIL import Image

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="F1 Analytics Task 2",
    page_icon="🏎️",
    layout="wide"
)

# --- ПУТИ К ФАЙЛАМ ---
# Так как мы в dashboard/2/app.py, то корень проекта на 2 уровня выше
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PROCESSOR_PATH = os.path.join(PROJECT_ROOT, "data_processor", "2", "task2_spark.py")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data_processor", "2", "reports")
PAIRS_CSV = os.path.join(REPORTS_DIR, "task2_pairs_summary.csv")
STYLE_CSV = os.path.join(PROJECT_ROOT, "data", "task2_driver_style.csv")
CORNERS_CSV = os.path.join(PROJECT_ROOT, "data", "task2_corners.csv")
LAPS_CSV = os.path.join(PROJECT_ROOT, "data", "task2_fastest_laps.csv")

# Переиспользуем функцию построения графика из модуля обработки
spec = importlib.util.spec_from_file_location("task2_spark", PROCESSOR_PATH)
task2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task2)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# --- ЗАГОЛОВОК ---
st.title("🏎️ Задача 2: Стиль вождения напарников по команде (Apache Spark)")
st.markdown(
    "Сравнение лучших кругов квалификации пилотов одной команды: скорость в поворотах, "
    "точки торможения и выхода на газ. Агрегация телеметрии выполнена в `PySpark DataFrame` (Window-функции).")

if not all(os.path.exists(p) for p in [STYLE_CSV, CORNERS_CSV, LAPS_CSV]):
    st.error("Данные не найдены. Сначала запустите `python data_processor/2/task2_spark.py` "
             "(или с флагом `--samples` для облегчённого сэмпла).")
    st.stop()

style_df = load_csv(STYLE_CSV)
corners_df = load_csv(CORNERS_CSV)
laps_df = load_csv(LAPS_CSV)

# --- ВКЛАДКИ ДАШБОРДА ---
tab1, tab2, tab3 = st.tabs(["🏁 Сравнение кругов", "📊 Итоги сезона", "🗂️ Метрики стиля"])

with tab1:
    st.header("Наложение телеметрии напарников")

    events = corners_df[["Year", "GrandPrix"]].drop_duplicates().sort_values(["Year", "GrandPrix"])
    event_labels = [f"{r.GrandPrix} {r.Year}" for r in events.itertuples()]
    col1, col2 = st.columns(2)
    with col1:
        event_idx = st.selectbox("Гран-при:", range(len(event_labels)), format_func=lambda i: event_labels[i])
    year, gp = events.iloc[event_idx]["Year"], events.iloc[event_idx]["GrandPrix"]
    event_corners = corners_df[(corners_df["Year"] == year) & (corners_df["GrandPrix"] == gp)]
    with col2:
        team = st.selectbox("Команда:", sorted(event_corners["Team"].unique()))

    team_corners = event_corners[event_corners["Team"] == team]
    d1, d2 = team_corners["Driver1"].iloc[0], team_corners["Driver2"].iloc[0]
    team_style = style_df[(style_df["Year"] == year) & (style_df["GrandPrix"] == gp)
                          & (style_df["Team"] == team)].set_index("Driver")

    # Ключевые цифры по кругу
    t1, t2 = team_style.loc[d1, "LapTime"], team_style.loc[d2, "LapTime"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Круг {d1}", f"{t1:.3f} с")
    m2.metric(f"Круг {d2}", f"{t2:.3f} с", delta=f"{t2 - t1:+.3f} с", delta_color="inverse")
    m3.metric("Поворотов найдено", len(team_corners))
    faster_corners = (team_corners["dTimeInCorner"] > 0).sum()
    m4.metric(f"Поворотов быстрее у {d1}", f"{faster_corners} из {len(team_corners)}")

    st.pyplot(task2.plot_overlay(laps_df, corners_df, year, gp, team))
    st.caption(f"Нижний график — накопленная разница во времени: выше нуля — {d1} впереди, ниже — {d2}.")

    st.subheader("Сравнение по поворотам")
    table = team_corners[["Corner", "ApexDistance", "CornerType",
                          "MinSpeed_1", "MinSpeed_2", "dMinSpeed",
                          "BrakeBeforeApex_1", "BrakeBeforeApex_2",
                          "ThrottleAfterApex_1", "ThrottleAfterApex_2", "dTimeInCorner"]].rename(columns={
        "Corner": "№", "ApexDistance": "Апекс, м", "CornerType": "Тип",
        "MinSpeed_1": f"Vmin {d1}", "MinSpeed_2": f"Vmin {d2}", "dMinSpeed": "ΔVmin, км/ч",
        "BrakeBeforeApex_1": f"Тормоз {d1}, м до апекса", "BrakeBeforeApex_2": f"Тормоз {d2}, м до апекса",
        "ThrottleAfterApex_1": f"Газ {d1}, м после", "ThrottleAfterApex_2": f"Газ {d2}, м после",
        "dTimeInCorner": f"Выигрыш {d1}, с",
    })


    # Подсветка: зелёный — преимущество Driver1, красный — Driver2
    def highlight_delta(val):
        if pd.isna(val) or val == 0: return ''
        return 'background-color: #2e7d32; color: white' if val > 0 else 'background-color: #c62828; color: white'


    st.dataframe(
        table.style.map(highlight_delta, subset=["ΔVmin, км/ч", f"Выигрыш {d1}, с"]).format(precision=1),
        width="stretch",
        height=400
    )

with tab2:
    st.header("Итоги по парам напарников")
    st.markdown("Все дельты считаются как **Driver1 − Driver2** (пилоты упорядочены по алфавиту).")

    images = [
        ("Разрыв в квалификации", "task2_quali_gap.png"),
        ("Скорость в поворотах по типам", "task2_corner_speed.png"),
        ("Работа педалями", "task2_brake_throttle.png"),
        ("Профиль стиля", "task2_style_heatmap.png"),
    ]
    for row in range(0, len(images), 2):
        cols = st.columns(2)
        for col, (title, filename) in zip(cols, images[row:row + 2]):
            with col:
                st.subheader(title)
                img_path = os.path.join(REPORTS_DIR, filename)
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), width="stretch")
                else:
                    st.warning("График не найден. Сначала запустите скрипт data_processor/2/task2_spark.py")

    st.divider()
    st.subheader("Сводная таблица")
    if os.path.exists(PAIRS_CSV):
        pairs_df = load_csv(PAIRS_CSV)
        st.dataframe(pairs_df.round(3), width="stretch")
        st.caption("MinSpeed_p_value — p-значение парного критерия Уилкоксона: "
                   "< 0.05 означает статистически значимую разницу скорости в поворотах.")
    else:
        st.info("Сводная таблица пока не сформирована.")

with tab3:
    st.header("Метрики стиля на лучшем круге (Spark-агрегация)")
    st.markdown("Доли считаются по пройденной дистанции круга: газ в пол (≥ 98%), торможение, "
                "накат (газ < 5% без тормоза); число зон торможения и переключений передач.")
    st.dataframe(style_df.round(2), width="stretch")
    st.markdown(f"**Всего записей:** {style_df.shape[0]} | **Признаков:** {style_df.shape[1]}")
