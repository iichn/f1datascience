import os
import re
import logging
import fastf1
import pandas as pd
from typing import Dict, Any


class F1DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self._setup_cache()

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        log_file = log_cfg.get("log_file", "logs/data_loader.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        log_format = log_cfg.get("format", "%(asctime)s [%(levelname)s]: %(message)s")

        self.logger = logging.getLogger("F1DataLoader")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Запись в файл
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)

        # Вывод в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)

    def _setup_cache(self):
        cache_dir = self.config["storage"]["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self.logger.info(f"FastF1 cache настроен: {cache_dir}")

    def _clean_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))

    def load_session_telemetry(self, year: int, grand_prix: Any, session_type: str = "R") -> pd.DataFrame:
        """
        Загрузка полной телеметрии сессии, обогащенной данными о командах,
        финишных позициях, возрасте шин и погоде.
        """
        self.logger.info(f"=== Загрузка сессии: {year} | Этап: {grand_prix} | {session_type} ===")

        try:
            session = fastf1.get_session(year, grand_prix, session_type)
            # Включаем weather=True для Задачи 5
            session.load(telemetry=True, laps=True, weather=True, messages=False)
        except Exception as e:
            self.logger.error(f"Ошибка выгрузки сессии {year} {grand_prix}: {e}")
            return pd.DataFrame()

        laps = session.laps
        if laps.empty:
            self.logger.warning(f"Данные по кругам отсутствуют для этапа {grand_prix}")
            return pd.DataFrame()

        gp_name = session.event['EventName'] if hasattr(session, 'event') else str(grand_prix)

        # 1. Формируем справочник результатов пилотов (для Задач 2 и 5)
        driver_meta = {}
        if hasattr(session, 'results') and not session.results.empty:
            for _, res in session.results.iterrows():
                abbr = res.get('Abbreviation')
                if abbr:
                    driver_meta[abbr] = {
                        'Team': res.get('TeamName', 'Unknown'),
                        'GridPosition': res.get('GridPosition', None),
                        'FinalPosition': res.get('Position', None),
                        'Status': res.get('Status', 'Unknown')
                    }

        # 2. Получаем усредненные показатели погоды (для Задачи 5)
        air_temp = None
        track_temp = None
        rainfall = False
        if hasattr(session, 'weather_data') and not session.weather_data.empty:
            air_temp = round(session.weather_data['AirTemp'].mean(), 1)
            track_temp = round(session.weather_data['TrackTemp'].mean(), 1)
            rainfall = bool(session.weather_data['Rainfall'].any())

        self.logger.info(f"Погода этапа: Воздух={air_temp}°C, Трасса={track_temp}°C, Дождь={rainfall}")

        telemetry_records = []
        self.logger.info(f"Обработка {len(laps)} кругов гонки...")

        for _, lap in laps.iterlaps():
            try:
                telem = lap.get_telemetry()
                if telem.empty:
                    continue

                driver = lap.get('Driver')
                meta = driver_meta.get(driver, {})

                # Обогащаем телеметрию метаданными под наши 5 задач:
                telem["Driver"] = driver
                telem["Team"] = meta.get('Team', lap.get('Team', 'Unknown'))  # Задача 2
                telem["LapNumber"] = lap.get('LapNumber')
                telem["LapTime_Sec"] = lap['LapTime'].total_seconds() if pd.notnull(lap.get('LapTime')) else None
                telem["Compound"] = lap.get('Compound')  # Задача 1
                telem["TyreLife"] = lap.get('TyreLife')  # Задача 1 (износ)
                telem["Stint"] = lap.get('Stint')  # Задача 1
                telem["TrackStatus"] = lap.get('TrackStatus')  # Задача 3 (аномалии)
                telem["GridPosition"] = meta.get('GridPosition')  # Задача 5
                telem["FinalPosition"] = meta.get('FinalPosition')  # Задача 5 (таргет)
                telem["AirTemp"] = air_temp  # Задача 5
                telem["TrackTemp"] = track_temp  # Задача 5
                telem["Rainfall"] = rainfall  # Задача 5
                telem["Year"] = year
                telem["GrandPrix"] = gp_name  # Задача 4
                telem["Session"] = session_type

                telemetry_records.append(telem)
            except Exception as e:
                self.logger.debug(f"Ошибка на круге {lap.get('LapNumber')}: {e}")
                continue

        if not telemetry_records:
            return pd.DataFrame()

        df = pd.concat(telemetry_records, ignore_index=True)
        self.logger.info(f"Готово: {len(df):,} строк с полным набором признаков.")
        return df

    def save_data(self, df: pd.DataFrame, filename: str, is_sample: bool = False):
        if df.empty:
            return

        target_dir = self.config["storage"]["samples_dir"] if is_sample else self.config["storage"]["output_dir"]
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, filename)
        tmp_path = target_path + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target_path)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        self.logger.info(f"Файл сохранен: {target_path} ({size_mb:.2f} MB)")

    def load_full_season(self, year: int, session_type: str = "R"):
        """Выгрузка всех 22 гонок сезона + создание сэмплов на 10000 строк."""
        self.logger.info(f"=== ЗАПУСК ВЫГРУЗКИ ВСЕХ ГОНОК СЕЗОНА {year} ===")

        # Получаем расписание
        schedule = fastf1.get_event_schedule(year)

        # Оставляем только официальные гонки (RoundNumber от 1 до 22)
        official_races = schedule[schedule['RoundNumber'] > 0]
        self.logger.info(f"Найдено официальных этапов для выгрузки: {len(official_races)}")

        output_dir = self.config["storage"]["output_dir"]
        samples_dir = self.config["storage"]["samples_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

        for _, event in official_races.iterrows():
            round_num = int(event['RoundNumber'])
            event_name = self._clean_name(event['EventName'])
            filename = f"{year}_Round_{round_num:02d}_{event_name}_{session_type}_telemetry.csv"
            target_path = os.path.join(output_dir, filename)

            sample_filename = f"sample_{filename}"
            target_sample_path = os.path.join(samples_dir, sample_filename)

            # Пропуск уже скачанных этапов
            if os.path.exists(target_path):
                # Если большой файл есть, а сэмпла к нему еще нет — быстро создаем сэмпл на 10000 строк
                if not os.path.exists(target_sample_path):
                    pd.read_csv(target_path, nrows=10000).to_csv(target_sample_path, index=False)
                    self.logger.info(f"Создан сэмпл (10000 строк) для готового файла: {sample_filename}")

                self.logger.info(
                    f"[Этап {round_num:02d}/{len(official_races)}] {event['EventName']} уже скачан, пропуск.")
                continue

            self.logger.info(
                f"\n>>> [Этап {round_num:02d}/{len(official_races)}] Начинаем сбор: {event['EventName']} <<<")

            # Загружаем полную телеметрию гонки
            df = self.load_session_telemetry(year=year, grand_prix=round_num, session_type=session_type)

            if not df.empty:
                # 1. Сохраняем полный многогигабайтный файл
                self.save_data(df, filename=filename, is_sample=False)
                # 2. Сохраняем легкий сэмпл ровно на 10000 строк
                self.save_data(df.head(10000), filename=sample_filename, is_sample=True)
            else:
                self.logger.warning(f"Не удалось получить данные для этапа {round_num:02d} ({event['EventName']})")