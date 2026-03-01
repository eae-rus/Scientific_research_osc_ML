"""
Модуль балансировки классов для обучения моделей классификации аварий.

Реализует две стратегии балансировки:
1. GlobalClassBalancer - глобальная балансировка по типам событий
2. OscillogramClassBalancer - балансировка внутри осциллограмм + глобальная

Предподготовка индексов выполняется один раз и кэшируется.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import polars as pl
from dataclasses import dataclass


@dataclass
class BalancingConfig:
    """Конфигурация балансировки."""
    min_batch_size: int = 64  # Минимальный размер батча
    samples_per_oscillogram: int = 12  # Мин. примеров из одной осциллограммы
    total_samples_per_epoch: int = 10000  # Целевое количество примеров в эпохе
    cache_dir: Optional[str] = None  # Директория для кэша


class ClassBalancerBase:
    """Базовый класс для балансировщиков."""
    
    def __init__(self, 
                 df: pl.DataFrame,
                 target_columns: List[str],
                 window_size: int = 320,
                 config: Optional[BalancingConfig] = None):
        """
        Args:
            df: DataFrame с данными (должен содержать 'file_name' и 'row_nr')
            target_columns: Список колонок с метками классов (multi-label)
            window_size: Размер окна для сэмплирования
            config: Конфигурация балансировки
        """
        self.df = df
        self.target_columns = target_columns
        self.window_size = window_size
        self.config = config or BalancingConfig()
        
        # Проверяем обязательные колонки
        required_cols = ['file_name', 'row_nr'] + target_columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        
        self._analysis_cache: Optional[Dict] = None
        
    def _compute_cache_hash(self) -> str:
        """Вычисляет хэш для кэширования на основе конфигурации и данных."""
        hash_str = f"{self.__class__.__name__}_{len(self.df)}_{self.window_size}_{','.join(self.target_columns)}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(self) -> Optional[Path]:
        """Возвращает путь к файлу кэша."""
        if self.config.cache_dir is None:
            return None
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"balancing_cache_{self._compute_cache_hash()}.json"
    
    def _load_cache(self) -> Optional[Dict]:
        """Загружает кэш если существует."""
        cache_path = self._get_cache_path()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_cache(self, data: Dict):
        """Сохраняет данные в кэш."""
        cache_path = self._get_cache_path()
        if cache_path:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
            except Exception:
                pass


class GlobalClassBalancer(ClassBalancerBase):
    """
    Глобальная балансировка по типам событий.
    
    Алгоритм:
    1. Анализирует весь датасет и группирует примеры по классам
    2. При сэмплировании берёт равное количество примеров из каждого класса
    3. Размер батча адаптируется под количество классов (минимум 64)
    """
    
    def __init__(self, 
                 df: pl.DataFrame,
                 target_columns: List[str],
                 window_size: int = 320,
                 config: Optional[BalancingConfig] = None):
        super().__init__(df, target_columns, window_size, config)
        self._class_indices: Optional[Dict[str, List[Tuple[int, int]]]] = None
        self._class_names: Optional[List[str]] = None
    
    def analyze(self) -> Dict[str, int]:
        """
        Анализирует датасет и строит индексы по классам.
        
        Returns:
            Словарь {class_name: count} с количеством примеров в каждом классе
        """
        if self._class_indices is not None:
            # Уже проанализировано
            return {name: len(indices) for name, indices in self._class_indices.items()}
        
        # Пробуем загрузить из кэша
        cached = self._load_cache()
        if cached:
            self._class_indices = {k: [tuple(x) for x in v] for k, v in cached['class_indices'].items()}
            self._class_names = cached['class_names']
            return cached['class_counts']
        
        print("[GlobalClassBalancer] Анализ датасета...")
        
        # Группируем по файлам
        file_stats = self.df.group_by("file_name").agg([
            pl.col("row_nr").min().alias("start_idx"),
            pl.len().alias("length")
        ]).sort("start_idx")
        
        # Строим индекс: для каждой точки определяем её класс
        # Класс определяется по комбинации активных меток (multi-label -> string)
        self._class_indices = {}
        
        for row in file_stats.iter_rows(named=True):
            file_name = row['file_name']
            start_idx = row['start_idx']
            length = row['length']
            
            # Пропускаем короткие файлы
            if length < self.window_size:
                continue
            
            # Извлекаем данные файла
            file_data = self.df.slice(start_idx, length)
            
            # Для каждой допустимой позиции окна определяем класс
            max_start = length - self.window_size
            for local_pos in range(0, max_start + 1, self.window_size // 4):  # Шаг 1/4 окна
                target_pos = local_pos + self.window_size - 1  # Метка в конце окна
                
                # Получаем метки для этой позиции
                labels = file_data.row(target_pos, named=True)
                active_labels = [col for col in self.target_columns if labels.get(col, 0) == 1]
                
                # Формируем ключ класса
                if not active_labels:
                    class_key = "Normal"
                else:
                    class_key = "_".join(sorted(active_labels))
                
                # Добавляем индекс (абсолютный start, length файла)
                if class_key not in self._class_indices:
                    self._class_indices[class_key] = []
                self._class_indices[class_key].append((start_idx + local_pos, self.window_size))
        
        self._class_names = sorted(self._class_indices.keys())
        
        # Сохраняем в кэш
        class_counts = {name: len(indices) for name, indices in self._class_indices.items()}
        self._save_cache({
            'class_indices': {k: v for k, v in self._class_indices.items()},
            'class_names': self._class_names,
            'class_counts': class_counts
        })
        
        print(f"[GlobalClassBalancer] Найдено {len(self._class_names)} классов:")
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  - {name}: {count} примеров")
        
        return class_counts
    
    def get_batch_config(self) -> Tuple[int, int]:
        """
        Вычисляет размер батча и количество шагов в эпохе.
        
        Returns:
            (batch_size, steps_per_epoch)
        """
        if self._class_indices is None:
            self.analyze()
        
        num_classes = len(self._class_names)
        
        # Размер батча должен делиться на количество классов
        # и быть не менее min_batch_size
        samples_per_class = max(1, self.config.min_batch_size // num_classes)
        
        # Корректируем если получилось меньше минимума
        if samples_per_class * num_classes < self.config.min_batch_size:
            samples_per_class = (self.config.min_batch_size + num_classes - 1) // num_classes
        
        batch_size = samples_per_class * num_classes
        
        # Количество шагов для достижения целевого количества примеров
        steps_per_epoch = self.config.total_samples_per_epoch // batch_size
        
        return batch_size, steps_per_epoch
    
    def sample_batch(self, rng: np.random.Generator = None) -> List[Tuple[int, int]]:
        """
        Сэмплирует один сбалансированный батч.
        
        Args:
            rng: Генератор случайных чисел (опционально)
            
        Returns:
            Список кортежей (start_idx, window_size) для каждого примера в батче
        """
        if self._class_indices is None:
            self.analyze()
        
        if rng is None:
            rng = np.random.default_rng()
        
        batch_size, _ = self.get_batch_config()
        samples_per_class = batch_size // len(self._class_names)
        
        batch = []
        for class_name in self._class_names:
            class_pool = self._class_indices[class_name]
            # Случайный выбор с возвратом (если примеров мало)
            indices = rng.choice(len(class_pool), size=samples_per_class, replace=True)
            for idx in indices:
                batch.append(class_pool[idx])
        
        # Перемешиваем батч
        rng.shuffle(batch)
        return batch
    
    def create_epoch_indices(self, rng: np.random.Generator = None) -> List[Tuple[int, int]]:
        """
        Создаёт индексы для одной эпохи обучения.
        
        Returns:
            Список всех индексов для эпохи
        """
        batch_size, steps = self.get_batch_config()
        
        if rng is None:
            rng = np.random.default_rng()
        
        all_indices = []
        for _ in range(steps):
            batch = self.sample_batch(rng)
            all_indices.extend(batch)
        
        return all_indices
    
    @property
    def num_classes(self) -> int:
        """Количество уникальных классов."""
        if self._class_names is None:
            self.analyze()
        return len(self._class_names)


class OscillogramClassBalancer(ClassBalancerBase):
    """
    Балансировка внутри осциллограмм + глобальная.
    
    Алгоритм:
    1. Для каждой осциллограммы определяет присутствующие классы
    2. Из каждой осциллограммы берёт равное количество примеров каждого класса
    3. Дополнительно применяет глобальную балансировку (pos_weight)
    
    Это предотвращает переобучение на осциллограммы с перекосом классов.
    """
    
    def __init__(self, 
                 df: pl.DataFrame,
                 target_columns: List[str],
                 window_size: int = 320,
                 config: Optional[BalancingConfig] = None):
        super().__init__(df, target_columns, window_size, config)
        self._osc_class_map: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._global_class_counts: Optional[Dict[str, int]] = None
        self._class_doc_counts: Optional[Dict[str, int]] = None  # В скольких файлах встречается класс

    def analyze(self) -> Dict[str, Dict]:
        """
        Анализирует датасет по осциллограммам.
        
        Returns:
            Словарь {file_name: {class_name: count}}
        """
        if self._osc_class_map is not None:
            return {fn: {cn: len(indices) for cn, indices in classes.items()} 
                    for fn, classes in self._osc_class_map.items()}
        
        # Пробуем загрузить из кэша
        cached = self._load_cache()
        if cached:
            self._osc_class_map = cached['osc_class_map']
            self._global_class_counts = cached['global_class_counts']
            self._class_doc_counts = cached.get('class_doc_counts', {})
            return {fn: {cn: len(indices) for cn, indices in classes.items()} 
                    for fn, classes in self._osc_class_map.items()}
        
        print("[OscillogramClassBalancer] Анализ датасета...")
        
        # Группируем по файлам
        file_stats = self.df.group_by("file_name").agg([
            pl.col("row_nr").min().alias("start_idx"),
            pl.len().alias("length")
        ]).sort("start_idx")
        
        self._osc_class_map = {}
        self._global_class_counts = {}
        self._class_doc_counts = {}
        
        for row in file_stats.iter_rows(named=True):
            file_name = row['file_name']
            start_idx = row['start_idx']
            length = row['length']
            
            if length < self.window_size:
                continue
            
            file_data = self.df.slice(start_idx, length)
            file_classes: Dict[str, List[int]] = {}
            
            max_start = length - self.window_size
            for local_pos in range(0, max_start + 1, self.window_size // 4):
                target_pos = local_pos + self.window_size - 1
                
                labels = file_data.row(target_pos, named=True)
                active_labels = [col for col in self.target_columns if labels.get(col, 0) == 1]
                
                if not active_labels:
                    class_key = "Normal"
                else:
                    class_key = "_".join(sorted(active_labels))
                
                if class_key not in file_classes:
                    file_classes[class_key] = []
                file_classes[class_key].append(start_idx + local_pos)
                
                # Глобальный подсчёт (общее количество примеров)
                self._global_class_counts[class_key] = self._global_class_counts.get(class_key, 0) + 1
            
            if file_classes:
                self._osc_class_map[file_name] = file_classes
                # Подсчет документной частоты (в скольких файлах есть этот класс)
                for class_key in file_classes.keys():
                    self._class_doc_counts[class_key] = self._class_doc_counts.get(class_key, 0) + 1
        
        # Сохраняем в кэш
        self._save_cache({
            'osc_class_map': self._osc_class_map,
            'global_class_counts': self._global_class_counts,
            'class_doc_counts': self._class_doc_counts
        })
        
        print(f"[OscillogramClassBalancer] Проанализировано {len(self._osc_class_map)} осциллограмм")
        print(f"  Встречаемость классов (кол-во файлов):")
        for name, count in sorted(self._class_doc_counts.items(), key=lambda x: -x[1]):
            print(f"  - {name}: {count} файлов")
        
        return {fn: {cn: len(indices) for cn, indices in classes.items()} 
                for fn, classes in self._osc_class_map.items()}
    
    def sample_from_oscillogram(self, file_name: str, rng: np.random.Generator = None) -> List[Tuple[int, int]]:
        """
        Сэмплирует примеры из одной осциллограммы с балансировкой классов.
        
        Returns:
            Список кортежей (start_idx, window_size)
        """
        if self._osc_class_map is None:
            self.analyze()
        
        if file_name not in self._osc_class_map:
            return []
        
        if rng is None:
            rng = np.random.default_rng()
        
        classes = self._osc_class_map[file_name]
        num_classes = len(classes)
        
        # Сколько примеров каждого класса брать
        min_samples = self.config.samples_per_oscillogram
        samples_per_class = max(1, min_samples // num_classes)
        
        # Если классов мало, увеличиваем samples_per_class чтобы набрать min_samples
        if samples_per_class * num_classes < min_samples:
            samples_per_class = (min_samples + num_classes - 1) // num_classes
        
        samples = []
        for class_name, indices in classes.items():
            chosen = rng.choice(len(indices), size=min(samples_per_class, len(indices)), replace=False)
            for idx in chosen:
                samples.append((indices[idx], self.window_size))
        
        return samples
    
    def create_epoch_indices(self, rng: np.random.Generator = None) -> List[Tuple[int, int]]:
        """
        Создаёт индексы для одной эпохи с балансировкой внутри осциллограмм.
        
        Returns:
            Список всех индексов для эпохи
        """
        if self._osc_class_map is None:
            self.analyze()
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Глобальная балансировка: выбираем осциллограммы пропорционально "редкости" их классов
        osc_weights = {}
        total_inverse = 0.0
        
        # Защита от деления на ноль, если class_doc_counts не заполнен
        if not self._class_doc_counts:
            # Fallback на глобальные каунты, если старый кэш или что-то пошло не так
            for class_name in self._global_class_counts:
                 self._class_doc_counts[class_name] = max(1, self._global_class_counts[class_name] // 12) # Грубая оценка

        for file_name, classes in self._osc_class_map.items():
            # Вес осциллограммы = сумма обратных документных частот её классов
            weight = 0.0
            for class_name in classes.keys():
                # Используем количество файлов, содержащих класс, а не количество примеров
                class_doc_freq = self._class_doc_counts.get(class_name, 1)
                weight += 1.0 / class_doc_freq
            osc_weights[file_name] = weight
            total_inverse += weight
        
        # Нормализуем веса
        osc_names = list(osc_weights.keys())
        # Проверка на total_inverse > 0
        if total_inverse <= 0:
             probs = None # Равномерное распределение
        else:
             probs = np.array([osc_weights[fn] / total_inverse for fn in osc_names])
        
        # Сколько осциллограмм нужно отобрать
        avg_samples_per_osc = self.config.samples_per_oscillogram
        num_osc_to_sample = self.config.total_samples_per_epoch // avg_samples_per_osc
        
        # Выбираем осциллограммы с учётом весов
        selected_osc = rng.choice(len(osc_names), size=num_osc_to_sample, replace=True, p=probs)
        
        all_indices = []
        for osc_idx in selected_osc:
            file_name = osc_names[osc_idx]
            samples = self.sample_from_oscillogram(file_name, rng)
            all_indices.extend(samples)
        
        # Перемешиваем
        rng.shuffle(all_indices)
        
        return all_indices
    
    def get_pos_weights(self) -> Dict[str, float]:
        """
        Вычисляет pos_weight для каждого класса (для BCEWithLogitsLoss).
        
        Returns:
            Словарь {class_name: weight}
        """
        if self._global_class_counts is None:
            self.analyze()
        
        total = sum(self._global_class_counts.values())
        weights = {}
        for class_name, count in self._global_class_counts.items():
            # pos_weight = (total - count) / count
            weights[class_name] = (total - count) / max(count, 1)
        
        return weights


def get_balancing_strategy(strategy: str,
                          df: pl.DataFrame,
                          target_columns: List[str],
                          window_size: int = 320,
                          config: Optional[BalancingConfig] = None) -> Optional[ClassBalancerBase]:
    """
    Фабричная функция для создания балансировщика.
    
    Args:
        strategy: Название стратегии ('global', 'oscillogram', 'none')
        df: DataFrame с данными
        target_columns: Колонки с метками
        window_size: Размер окна
        config: Конфигурация
        
    Returns:
        Экземпляр балансировщика или None для 'none'
    """
    if strategy == 'global':
        return GlobalClassBalancer(df, target_columns, window_size, config)
    elif strategy == 'oscillogram':
        return OscillogramClassBalancer(df, target_columns, window_size, config)
    elif strategy == 'none':
        return None
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}. Use 'global', 'oscillogram', or 'none'.")
