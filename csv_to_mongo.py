import os
import glob
import pandas as pd
from pymongo import MongoClient

# --- НАСТРОЙКИ ---
DATA_FOLDER = "data/raw_telemetry"  # Папка с CSV-файлами
DB_NAME = "f1_database"  # Имя базы данных
COLLECTION_NAME = "telemetry"  # Имя коллекции (таблицы)


def import_telemetry_to_mongo():
    print("1. Подключение к MongoDB...")
    # Подключаемся к локальной базе данных
    client = MongoClient("mongodb://localhost:27017/")
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Формируем путь к папке с данными
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, DATA_FOLDER)

    # Ищем все CSV файлы по указанному пути
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

    if not csv_files:
        print(f"В папке {target_dir} не найдено ни одного CSV файла.")
        return

    print(f"Найдено файлов: {len(csv_files)}. Начинаем перенос в MongoDB...\n")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Чтение файла {filename}...")

        # Читаем данные через Pandas
        df = pd.read_csv(file_path)

        if df.empty:
            print(f"Файл {filename} пуст, пропускаем.")
            continue

        # Заменяем Pandas-специфичные NaN/NaT на стандартный питоновский None (чтобы MongoDB их поняла)
        df = df.where(pd.notnull(df), None)

        # Конвертируем DataFrame в список словарей (формат документов MongoDB)
        records = df.to_dict(orient="records")

        # Массовая вставка документов в коллекцию
        collection.insert_many(records)
        print(f"Загружено {len(records)} строк.")

    print("\nВсе файлы успешно загружены!")

    # Создаем индексы для того, чтобы база мгновенно отвечала на запросы по году и Гран-при
    print("\nСоздание индексов для быстрого поиска...")
    collection.create_index([
        ("Year", 1),
        ("Session", 1),
        ("GrandPrix", 1)
    ])
    print("Составной индекс (Year, Session, GrandPrix) успешно создан.")

    # Выводим итоговую статистику
    total_docs = collection.count_documents({})
    print(f"\n Всего документов в коллекции '{COLLECTION_NAME}': {total_docs:,}")


if __name__ == "__main__":
    # Запускаем функцию загрузки
    import_telemetry_to_mongo()