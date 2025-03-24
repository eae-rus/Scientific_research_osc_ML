# 1) open 1 test osc
# 2) make csv
import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Импортируем необходимые классы
from dataflow.comtrade_processing import ReadComtrade
from dataflow.raw_to_csv import RawToCSV


def main():
    # Определяем пути к директориям
    raw_path = os.path.join(project_root, 'data', 'raw')
    csv_path = os.path.join(project_root, 'data', 'csv')

    # Создаем директорию для выходных CSV-файлов, если она не существует
    os.makedirs(csv_path, exist_ok=True)

    # Инициализируем класс для конвертации Comtrade в CSV
    converter = RawToCSV(raw_path=raw_path, csv_path=csv_path)

    # Проверка наличия файлов в директории
    cfg_files = [f for f in os.listdir(raw_path) if f.endswith('.cfg')]
    if not cfg_files:
        print(f"Ошибка: В директории {raw_path} не найдены .cfg файлы.")
        return

    print(f"Найдено {len(cfg_files)} файлов .cfg в директории {raw_path}")

    try:
        # Запускаем конвертацию
        print("Начинаем конвертацию Comtrade в CSV...")
        output_df = converter.create_csv(csv_name='output.csv')

        # Выводим информацию о результатах
        print(f"Конвертация завершена успешно.")
        print(f"Размер выходного DataFrame: {output_df.shape}")
        print(f"CSV-файл сохранен в: {os.path.join(csv_path, 'output.csv')}")

        # Выводим первые несколько строк для проверки
        print("\nПервые 5 строк результата:")
        print(output_df.head())

    except Exception as e:
        print(f"Ошибка при конвертации: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()