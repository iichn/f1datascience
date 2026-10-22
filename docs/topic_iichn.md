# 20.10.2026

создать окружение:
py -m venv .venv

активировать:
.venv\Scripts\Activate.ps1

обновить пип и устновить зависимости:
python -m pip install --upgrade pip
pip install -r requirements.txt

выгрузить csv файлы в папку data/raw_telemetry:
py bin/run_loader.py --year 2022 --session R --all-racec

сейчас решено выгрузить только 3 гонки для теста (семплы не подходят т.к. в квалификациях износ шин не играет роли)
py bin/run_loader.py --year 2022 --gp Bahrain --session R
py bin/run_loader.py --year 2022 --gp Monza --session R
py bin/run_loader.py --year 2022 --gp Monaco --session R

## Задача
Оценка темпа износа шин и анализ влияния пройденной дистанции на падение скорости автомобиля по ходу заезда.

Технически Spark тут не нужен. Агрегированные круги за весь сезон это ~1,2 тыс. кругов на гонку, десятки тысяч строк. pandas справляется за секунды. На трёх гонках Spark даже медленнее из-за старта JVM.
Компромисс: тяжёлую часть (сырая телеметрия → круги, там миллионы строк) делаем на Spark, а подгонку регрессии по стинтам оставляем на pandas.

### запуск анализа:
py -m data_processor.task1_iichn.run_tyre_wear

### запуск дашборда:
streamlit run dashboard/1/dashboard.py

#### у меня стоит джава 8. чтобы анализ сработал нормально можно перед запуском прописать:
$env:JAVA_HOME = (Get-ChildItem "C:\Program Files\Eclipse Adoptium\jdk-17*" | Select-Object -First 1).FullName
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
java -version