import argparse
import sys
import os
import yaml

# Добавляем корень проекта в путь поиска модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader.loader import F1DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="F1 Data Ingestion Module (Big Data Coursework)")
    parser.add_argument(
        "-c", "--config",
        default="config/data_loader.yaml",
        help="Путь к конфигурационному YAML файлу"
    )
    parser.add_argument("--year", type=int, default=2023, help="Год сезона Formula 1")
    parser.add_argument("--gp", type=str, default="Monza", help="Название Гран-при (например, Monza, Monaco)")
    parser.add_argument("--session", type=str, default="Q", choices=["R", "Q", "FP1", "FP2", "FP3"], help="Тип сессии")
    parser.add_argument("--sample", action="store_true", help="Сохранить результат как эталонный срез в папку samples/")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"Ошибка: Конфигурационный файл {args.config} не найден!")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = F1DataLoader(config)
    loader.logger.info("Запуск модуля сбора данных через CLI...")

    # Выгрузка указанной сессии
    df = loader.load_session_telemetry(year=args.year, grand_prix=args.gp, session_type=args.session)

    # Имя файла
    file_prefix = "sample_" if args.sample else ""
    filename = f"{file_prefix}{args.year}_{args.gp}_{args.session}_telemetry.csv"

    loader.save_data(df, filename=filename, is_sample=args.sample)
    loader.logger.info("Сбор успешно завершен!")

if __name__ == "__main__":
    main()