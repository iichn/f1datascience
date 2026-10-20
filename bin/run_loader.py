import argparse
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader.loader import F1DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Модуль выгрузки телеметрии Formula 1")
    parser.add_argument("-c", "--config", default="config/data_loader.yaml", help="Путь к конфигу")
    parser.add_argument("--year", type=int, default=2022, help="Сезон чемпионата")
    parser.add_argument("--gp", type=str, default="Monza", help="Название Гран-при (для одного этапа)")
    parser.add_argument("--session", type=str, default="R", choices=["R", "Q", "FP1", "FP2", "FP3"], help="Тип сессии")
    parser.add_argument("--all-races", action="store_true", help="Выгрузить ВСЕ этапы сезона")
    parser.add_argument("--sample", action="store_true", help="Сохранить как сэмпл в папку samples/")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print(f"Ошибка: Конфиг {args.config} не найден!")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = F1DataLoader(config)

    if args.all_races:
        loader.load_full_season(year=args.year, session_type=args.session)
    else:
        df = loader.load_session_telemetry(year=args.year, grand_prix=args.gp, session_type=args.session)
        file_prefix = "sample_" if args.sample else ""
        filename = f"{file_prefix}{args.year}_{args.gp}_{args.session}_telemetry.csv"
        loader.save_data(df, filename=filename, is_sample=args.sample)

    loader.logger.info("Работа скрипта завершена.")

if __name__ == "__main__":
    main()