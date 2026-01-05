import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm
import sys

# Добавляем корневую директорию в путь
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from osc_tools.ml.dataset import OscillogramDataset

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scripts/stats_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_stats(csv_path: str, output_path: str, max_files: int = None, window_size: int = 1, physical_normalization: bool = True, norm_coef_path: str = None):
    """
    Рассчитывает глобальные статистики (mean, std) для датасета, представленного в виде CSV.
    CSV содержит объединенные данные осциллограмм.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"Dataset CSV not found: {csv_path}")
        return

    logger.info(f"Загрузка датасета из {csv_path}...")
    # Читаем CSV. Если файл очень большой, можно использовать scan_csv, но для 50-500МБ read_csv ок.
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Не удалось прочитать CSV: {e}")
        return

    logger.info(f"Загружен DataFrame размером {df.shape}")

    # Получаем список уникальных файлов
    if 'file_name' not in df.columns:
        logger.error("Колонка 'file_name' не найдена в CSV.")
        return

    file_names = df['file_name'].unique().to_list()
    logger.info(f"Найдено {len(file_names)} уникальных осциллограмм.")

    if max_files:
        file_names = file_names[:max_files]
        logger.info(f"Обработка первых {max_files} файлов.")

    # Инициализация аккумуляторов
    # current: IA, IB, IC, In
    # voltage: UA, UB, UC, Un
    stats = {
        'current': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0},
        'voltage': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0}
    }

    # Dummy dataset for methods
    if norm_coef_path is None:
        norm_coef_path = PROJECT_ROOT / 'data' / 'ml_datasets' / 'norm_coef_all_v1.4.csv'
    dummy_ds = OscillogramDataset(pl.DataFrame(), [], window_size=window_size, physical_normalization=physical_normalization, norm_coef_path=str(norm_coef_path))

    processed_count = 0
    
    # Группируем по файлам и обрабатываем
    # Чтобы не делать filter на каждой итерации (медленно), лучше разбить датафрейм
    # partition_by работает хорошо
    
    # Фильтруем только нужные файлы если max_files задан
    if max_files:
        df = df.filter(pl.col('file_name').is_in(file_names))

    partitions = df.partition_by("file_name", as_dict=True)

    for fname, file_df in tqdm(partitions.items(), desc="Расчет статистик"):
        try:
            # Применяем физическую нормализацию, если включена
            if physical_normalization:
                file_df = dummy_ds._apply_physical_normalization(file_df, fname)
            
            # Используем Smart Selector
            # [IA, IB, IC, In, UA, UB, UC, Un]
            standardized_data = dummy_ds._get_standardized_raw_data(file_df)
            
            # standardized_data shape: (Time, 8)
            currents = standardized_data[:, 0:4].flatten()
            voltages = standardized_data[:, 4:8].flatten()
            
            # Update stats
            stats['current']['count'] += currents.size
            stats['current']['sum'] += np.sum(currents)
            stats['current']['sum_sq'] += np.sum(currents**2)
            
            stats['voltage']['count'] += voltages.size
            stats['voltage']['sum'] += np.sum(voltages)
            stats['voltage']['sum_sq'] += np.sum(voltages**2)
            
            processed_count += 1

        except Exception as e:
            logger.error(f"Ошибка обработки {fname}: {e}")

    logger.info(f"Обработано {processed_count} файлов.")

    # Final calculation
    results = {}
    for key in ['current', 'voltage']:
        n = stats[key]['count']
        if n > 0:
            mean = stats[key]['sum'] / n
            variance = (stats[key]['sum_sq'] / n) - (mean ** 2)
            std = np.sqrt(variance)
            results[key] = {
                'mean': float(mean),
                'std': float(std),
                'count': int(n)
            }
        else:
            results[key] = {'mean': 0.0, 'std': 1.0, 'count': 0}

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Статистика сохранена в {output_path}")
    logger.info(f"Результаты: {json.dumps(results, indent=4)}")

if __name__ == "__main__":
    # Константы для расчета статистик
    CSV_PATH = "data/ml_datasets/labeled_2025_12_03.csv"
    OUTPUT_PATH = "data/ml_datasets/dataset_stats.json"
    MAX_FILES = None  # None для обработки всех файлов
    WINDOW_SIZE = 1
    PHYSICAL_NORMALIZATION = True
    NORM_COEF_PATH = None  # Используется по умолчанию: raw_data/norm_coef_all_v1.4.csv
    
    calculate_stats(CSV_PATH, OUTPUT_PATH, MAX_FILES, WINDOW_SIZE, PHYSICAL_NORMALIZATION, NORM_COEF_PATH)
