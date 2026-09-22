import os
import fastf1

# 1. Создаем папку под кеш, если ее еще нет
os.makedirs('f1_cache', exist_ok=True)

# 2. Включаем кеширование
fastf1.Cache.enable_cache('f1_cache')

# Загружаем квалификацию этапа в Монце 2023
session = fastf1.get_session(2023, 'Monza', 'Q')
session.load()

# Берем быстрейший круг сессии
fastest_lap = session.laps.pick_fastest()

# Получаем высокочастотную телеметрию датчиков
telemetry = fastest_lap.get_telemetry()

print(f"\nПилот: {fastest_lap['Driver']}")
print(f"Время круга: {fastest_lap['LapTime']}")
print(f"Количество замеров датчиков за один круг: {len(telemetry)}")
print("\nПервые 5 строк с показаниями датчиков болида:")
print(telemetry[['Date', 'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z']].head())