"""
Формирование облегчённого сэмпла для Задачи 2.

Обычные сэмплы (первые 10000 строк сессии) содержат телеметрию только одного пилота,
поэтому сравнить напарников по ним нельзя. Здесь для нескольких квалификаций
сохраняются лучшие круги ВСЕХ пилотов — этого достаточно для запуска
task2_spark.py --samples и дашборда без полной выгрузки.
"""
import os
import sys
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # пути в конфиге загрузчика заданы относительно корня проекта

from data_loader.loader import F1DataLoader

YEAR = 2022
# Этапы с разным профилем трассы: Бахрейн (торможения), Испания (универсальная), Монца (скоростная)
ROUNDS = [1, 6, 16]
KEEP_COLS = ["Year", "GrandPrix", "Session", "Team", "Driver", "LapNumber", "LapTime_Sec",
             "Distance", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS", "X", "Y"]


def main():
    with open("config/data_loader.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    loader = F1DataLoader(config)

    for round_num in ROUNDS:
        df = loader.load_session_telemetry(year=YEAR, grand_prix=round_num, session_type="Q")
        if df.empty:
            loader.logger.warning(f"Нет данных для этапа {round_num}")
            continue

        # Оставляем только лучший круг каждого пилота
        best = df.dropna(subset=["LapTime_Sec"]) \
            .groupby("Driver")[["LapNumber", "LapTime_Sec"]] \
            .apply(lambda g: g.loc[g["LapTime_Sec"].idxmin(), "LapNumber"]) \
            .rename("BestLap").reset_index()
        df = df.merge(best, on="Driver")
        df = df[df["LapNumber"] == df["BestLap"]][KEEP_COLS].round(3)

        gp_name = loader._clean_name(df["GrandPrix"].iloc[0])
        filename = f"sample_task2_{YEAR}_Round_{round_num:02d}_{gp_name}_Q_telemetry.csv"
        loader.save_data(df, filename=filename, is_sample=True)

    loader.logger.info("Сэмпл для Задачи 2 сформирован.")


if __name__ == "__main__":
    main()
