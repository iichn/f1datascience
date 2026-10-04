import os
import logging
import fastf1
import pandas as pd
from typing import List, Dict, Any

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

        # Настраиваем логгер: пишет и в файл, и в консоль
        self.logger = logging.getLogger("F1DataLoader")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Файловый хэндлер
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)

        # Консольный хэндлер для наблюдения процесса
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)

    def _setup_cache(self):
        cache_dir = self.config["storage"]["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self.logger.info(f"FastF1 cache инициализирован в: {cache_dir}")

    def load_session_telemetry(self, year: int, grand_prix: str, session_type: str) -> pd.DataFrame:
        """Загрузка полной телеметрии одной сессии."""
        self.logger.info(f"Загрузка сессии: {year} - {grand_prix} ({session_type})")
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(telemetry=True, laps=True, weather=True)

        laps = session.laps
        if laps.empty:
            self.logger.warning(f"Данные кругов пусты для {grand_prix} {session_type}")
            return pd.DataFrame()

        telemetry_records = []
        for _, lap in laps.iterlaps():
            try:
                telem = lap.get_telemetry()
                if not telem.empty:
                    telem["Driver"] = lap["Driver"]
                    telem["LapNumber"] = lap["LapNumber"]
                    telem["Compound"] = lap["Compound"]
                    telem["Stint"] = lap["Stint"]
                    telem["Year"] = year
                    telem["GrandPrix"] = grand_prix
                    telem["Session"] = session_type
                    telemetry_records.append(telem)
            except Exception as e:
                self.logger.debug(f"Пропуск круга {lap.get('LapNumber')}: {e}")
                continue

        if not telemetry_records:
            return pd.DataFrame()

        df = pd.concat(telemetry_records, ignore_index=True)
        self.logger.info(f"Успешно обработано строк телеметрии: {len(df)}")
        return df

    def save_data(self, df: pd.DataFrame, filename: str, is_sample: bool = False):
        if df.empty:
            return

        target_dir = self.config["storage"]["samples_dir"] if is_sample else self.config["storage"]["output_dir"]
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, filename)
        df.to_csv(target_path, index=False)
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        self.logger.info(f"Файл сохранен: {target_path} ({size_mb:.2f} MB)")