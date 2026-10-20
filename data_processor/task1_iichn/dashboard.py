import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processor.task1_iichn.tyre_wear import degradation

DATA = os.path.join(ROOT, "data", "processed", "task1_iichn", "laps_clean.csv")
COLORS = {"SOFT": "#e10600", "MEDIUM": "#f5c400", "HARD": "#bdbdbd"}

st.set_page_config(page_title="Задача 1: износ шин", layout="wide")
st.title("Износ шин и падение скорости")


@st.cache_data
def load():
    return pd.read_csv(DATA)


laps = load()

gps = st.sidebar.multiselect("Гран-при", sorted(laps.GrandPrix.unique()), default=sorted(laps.GrandPrix.unique()))
drivers = st.sidebar.multiselect("Пилоты (пусто = все)", sorted(laps.Driver.unique()))
fuel_k = st.sidebar.slider("Поправка на топливо, с/круг", 0.0, 0.10, 0.055, 0.005)

d = laps[laps.GrandPrix.isin(gps)]
if drivers:
    d = d[d.Driver.isin(drivers)]
d = d.assign(LapTimeFC=d.LapTime + fuel_k * (d.LapNumber - 1))

deg = degradation(d, fuel_k)
if deg.empty:
    st.warning("Недостаточно данных для выбранных фильтров")
    st.stop()

med = deg.groupby("Compound")[["DegSecPerLap", "SpeedKmhPerLap"]].median()
cols = st.columns(len(med))
for col, (comp, row) in zip(cols, med.iterrows()):
    col.metric(f"{comp}: потеря темпа", f"{row.DegSecPerLap:.3f} с/круг", f"{row.SpeedKmhPerLap:.2f} км/ч за круг", delta_color="off")

left, right = st.columns(2)
left.plotly_chart(px.scatter(d, x="TyreLife", y="LapTimeFC", color="Compound", color_discrete_map=COLORS,
                             opacity=0.6, title="Время круга (с поправкой на топливо) vs возраст шин"), use_container_width=True)
right.plotly_chart(px.scatter(d, x="TyreLife", y="AvgSpeed", color="Compound", color_discrete_map=COLORS,
                              opacity=0.6, title="Средняя скорость круга vs возраст шин"), use_container_width=True)

st.plotly_chart(px.box(deg, x="Compound", y="DegSecPerLap", color="Compound", color_discrete_map=COLORS,
                       points="all", title="Деградация по стинтам, с/круг"), use_container_width=True)

with st.expander("Таблица по стинтам"):
    st.dataframe(deg.round(3), use_container_width=True)