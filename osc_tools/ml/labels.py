import polars as pl
from typing import List, Dict, Optional

def get_ml_columns(df: pl.DataFrame) -> List[str]:
    """Возвращает все имена колонок, начинающиеся с 'ML_'."""
    return [c for c in df.columns if c.startswith("ML_")]

def clean_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Очищает колонки меток ML:
    - Приводит к типу Float32
    - Заполняет `null` значением 0.0
    - Приводит к Int8 (0/1)
    """
    ml_cols = get_ml_columns(df)
    if not ml_cols:
        return df
        
    exprs = []
    for col in ml_cols:
        # Обработка строковых представлений '1.0', '0.0' и None
        # Сначала приводим к Float32 (корректирует '1.0'), затем заполняем null=0, затем кастим в Int8
        exprs.append(
            pl.col(col).cast(pl.Float32, strict=False).fill_null(0.0).cast(pl.Int8).alias(col)
        )
        
    return df.with_columns(exprs)

def add_base_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавляет базовые колонки меток:
    - `Target_ML_1`: любая метка `ML_1*`
    - `Target_ML_2`: любая метка `ML_2*`
    - `Target_ML_3`: любая метка `ML_3*`
    - `Target_Normal`: ни одна из вышеперечисленных
    """
    ml_cols = get_ml_columns(df)
    
    # Метки по уровням
    ml_1_cols = [c for c in ml_cols if c.startswith("ML_1")]
    ml_2_cols = [c for c in ml_cols if c.startswith("ML_2")]
    ml_3_cols = [c for c in ml_cols if c.startswith("ML_3")]
    
    exprs = []
    
    # Вспомогательная функция: логическое OR (максимум по строке)
    def make_or_expr(cols, alias):
        if not cols:
            return pl.lit(0, dtype=pl.Int8).alias(alias)
        # max() по колонкам для каждой строки
        return pl.max_horizontal(cols).fill_null(0).cast(pl.Int8).alias(alias)

    exprs.append(make_or_expr(ml_1_cols, "Target_ML_1"))
    exprs.append(make_or_expr(ml_2_cols, "Target_ML_2"))
    exprs.append(make_or_expr(ml_3_cols, "Target_ML_3"))
    
    df = df.with_columns(exprs)
    
    # Target_Normal: 1, если все остальные равны 0
    df = df.with_columns(
        (
            (pl.col("Target_ML_1") == 0) & 
            (pl.col("Target_ML_2") == 0) & 
            (pl.col("Target_ML_3") == 0)
        ).cast(pl.Int8).alias("Target_Normal")
    )
    
    return df


def add_sequential_base_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавляет базовые метки для последовательных голов:
    - Target_Normal: нормальный режим (нет событий)
    - Target_ML_1: любая метка ML_1*
    - Target_ML_3: любая метка ML_3*
    - Target_ML_2: любая метка ML_2*, но если Target_ML_3=1, то Target_ML_2=0
    """
    # Сначала создаём стандартные базовые метки
    df = add_base_labels(df)

    # Если есть авария (ML_3), то ML_2 должен быть 0
    df = df.with_columns(
        pl.when(pl.col("Target_ML_3") == 1)
        .then(0)
        .otherwise(pl.col("Target_ML_2"))
        .cast(pl.Int8)
        .alias("Target_ML_2")
    )

    # Пересчитываем Target_Normal с учётом коррекции
    df = df.with_columns(
        (
            (pl.col("Target_ML_1") == 0) &
            (pl.col("Target_ML_2") == 0) &
            (pl.col("Target_ML_3") == 0)
        ).cast(pl.Int8).alias("Target_Normal")
    )

    return df


def propagate_hierarchical_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Распространяет метки вверх по иерархии (full_by_levels).
    
    Если ML_2_3_1 = 1, то:
    - ML_2_3 должен быть = 1
    - ML_2 должен быть = 1
    
    Это гарантирует согласованность иерархических меток.
    
    Returns:
        DataFrame с исправленными иерархическими метками
    """
    ml_cols = get_ml_columns(df)
    if not ml_cols:
        return df
    
    # Группируем колонки по уровням глубины (количество подчеркиваний)
    # ML_1 -> depth 1
    # ML_1_1 -> depth 2  
    # ML_1_1_1 -> depth 3
    
    # Строим словарь: родитель -> [дочерние колонки]
    parent_children: Dict[str, List[str]] = {}
    
    for col in ml_cols:
        parts = col.split('_')
        if len(parts) > 2:  # Есть родитель (ML_X_Y -> родитель ML_X)
            parent = '_'.join(parts[:-1])
            if parent in ml_cols:
                if parent not in parent_children:
                    parent_children[parent] = []
                parent_children[parent].append(col)
    
    # Проходим от самых вложенных уровней к корню
    # Сортируем родителей по глубине (количеству '_'), от глубоких к верхним
    sorted_parents = sorted(parent_children.keys(), key=lambda x: -x.count('_'))
    
    for parent in sorted_parents:
        children = parent_children[parent]
        if children:
            # Родитель = max(родитель, max(дети))
            # Если хотя бы один ребенок = 1, родитель должен быть = 1
            df = df.with_columns(
                pl.max_horizontal([pl.col(parent)] + [pl.col(c) for c in children])
                .fill_null(0)
                .cast(pl.Int8)
                .alias(parent)
            )
    
    return df


def add_intermediate_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Добавляет промежуточные уровни меток для иерархической классификации.
    
    Создает колонки:
    - Target_Level1_ML_X: метки первого уровня (ML_1, ML_2, ML_3)
    - Target_Level2_ML_X_Y: метки второго уровня (ML_1_1, ML_2_1, ...)
    
    Это полезно для многоуровневой оценки качества модели.
    """
    ml_cols = get_ml_columns(df)
    
    # Уровень 1: ML_1, ML_2, ML_3 (уже есть в base_labels как Target_ML_X)
    # Уровень 2: ML_X_Y (например, ML_2_1, ML_2_3)
    
    level2_cols = [c for c in ml_cols if c.count('_') == 2]  # ML_X_Y
    
    # Для каждой колонки уровня 2 создаем Target_Level2_* 
    # (суммируя её детей, если они есть)
    exprs = []
    for col in level2_cols:
        # Находим дочерние колонки (ML_X_Y_Z)
        children = [c for c in ml_cols if c.startswith(col + '_')]
        
        if children:
            # Если есть дети, берем max(колонка, дети)
            expr = pl.max_horizontal([pl.col(col)] + [pl.col(c) for c in children])
        else:
            expr = pl.col(col)
            
        exprs.append(expr.fill_null(0).cast(pl.Int8).alias(f"Target_Level2_{col}"))
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df


def get_target_columns(level: str = 'base', df: Optional[pl.DataFrame] = None) -> List[str]:
    """
    Возвращает список целевых колонок для указанного уровня.

    Args:
        level: 
            - 'base' или 'base_labels': 4 обобщённых класса (Target_Normal, Target_ML_1, Target_ML_2, Target_ML_3)
            - 'base_sequential': 4 класса (Target_Normal, Target_ML_1, Target_ML_2, Target_ML_3) с ограничением ML_2=0 при ML_3=1
            - 'full': все колонки ML_* (требует df)
            - 'full_by_levels': все ML_* колонки с распространением иерархии (требует df)
            - 'level1': только колонки первого уровня (ML_1, ML_2, ML_3)
            - 'level2': колонки второго уровня (ML_X_Y)
        df: DataFrame (требуется для level='full' или 'full_by_levels')
        
    Returns:
        Список имён целевых колонок
    """
    if level in ('base', 'base_labels'):
        return ["Target_Normal", "Target_ML_1", "Target_ML_2", "Target_ML_3"]
    elif level == 'base_sequential':
        return ["Target_Normal", "Target_ML_1", "Target_ML_2", "Target_ML_3"]
    
    elif level == 'full':
        if df is None:
            raise ValueError("Для уровня 'full' необходимо передать DataFrame")
        return get_ml_columns(df)
    
    elif level == 'full_by_levels':
        if df is None:
            raise ValueError("Для уровня 'full_by_levels' необходимо передать DataFrame")
        # Возвращаем все ML_* колонки (данные уже должны быть обработаны propagate_hierarchical_labels)
        return get_ml_columns(df)
    
    elif level == 'level1':
        # Только верхний уровень: ML_1, ML_2, ML_3
        if df is not None:
            return [c for c in get_ml_columns(df) if c.count('_') == 1]
        return ["ML_1", "ML_2", "ML_3"]
    
    elif level == 'level2':
        if df is None:
            raise ValueError("Для уровня 'level2' необходимо передать DataFrame")
        return [c for c in get_ml_columns(df) if c.count('_') == 2]
    
    elif level == 'ozz':
        # 3 целевых класса задачи ОЗЗ/ДПОЗЗ (multi-label)
        return ["Target_OZZ", "Target_OZZ_decay", "Target_OZZ_dpozz"]
    
    else:
        raise ValueError(f"Неизвестный уровень: {level}. Допустимые: 'base', 'base_labels', 'full', 'full_by_levels', 'level1', 'level2', 'ozz'")


def prepare_labels_for_experiment(
    df: pl.DataFrame, 
    target_level: str = 'base'
) -> pl.DataFrame:
    """
    Подготавливает метки DataFrame для эксперимента с заданным уровнем детализации.
    
    Args:
        df: Исходный DataFrame
        target_level: Уровень детализации меток
        
    Returns:
        DataFrame с подготовленными метками
    """
    # Всегда очищаем ML колонки
    df = clean_labels(df)
    
    if target_level in ('base', 'base_labels'):
        # Добавляем базовые метки (4 класса)
        df = add_base_labels(df)
    elif target_level == 'base_sequential':
        # Добавляем базовые метки с ограничением ML_2=0 при ML_3=1
        df = add_sequential_base_labels(df)
        
    elif target_level == 'full_by_levels':
        # Распространяем метки по иерархии
        df = propagate_hierarchical_labels(df)
        # Также добавляем базовые для возможности сравнения
        df = add_base_labels(df)
        
    elif target_level == 'full':
        # Просто используем все ML_* как есть
        # Но добавляем базовые для Hierarchical Accuracy
        df = add_base_labels(df)
    
    elif target_level == 'ozz':
        # Специализированные метки для задачи ОЗЗ/ДПОЗЗ (3 класса)
        from osc_tools.data_management.ozz_split import add_ozz_target_columns
        df = add_ozz_target_columns(df)
        
    return df


def get_label_hierarchy() -> Dict[str, List[str]]:
    """
    Возвращает структуру иерархии меток для визуализации и анализа.
    
    Это статическая структура, основанная на известной схеме меток проекта.
    
    Returns:
        Словарь {родительская_метка: [дочерние_метки]}
    """
    return {
        'ML_1': ['ML_1_1', 'ML_1_2'],
        'ML_1_1': ['ML_1_1_1'],
        'ML_2': ['ML_2_1', 'ML_2_2', 'ML_2_3', 'ML_2_4', 'ML_2_5_1', 'ML_2_6', 'ML_2_7_1', 'ML_2_7_2'],
        'ML_2_1': ['ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3'],
        'ML_2_3': ['ML_2_3_1'],
        'ML_2_4': ['ML_2_4_1', 'ML_2_4_2'],
        'ML_3': ['ML_3_1', 'ML_3_2', 'ML_3_3', 'ML_3_4', 'ML_3_5', 'ML_3_6'],
        'ML_3_3': ['ML_3_3_2', 'ML_3_3_3'],
        'ML_3_4': ['ML_3_4_1', 'ML_3_4_2'],
        'ML_3_6': ['ML_3_6_1']
    }


def count_labels_per_level(df: pl.DataFrame) -> Dict[str, int]:
    """
    Подсчитывает количество меток на каждом уровне иерархии.
    
    Returns:
        Словарь {уровень: количество_меток}
    """
    ml_cols = get_ml_columns(df)
    
    level_counts = {
        'level1': 0,  # ML_X
        'level2': 0,  # ML_X_Y
        'level3': 0,  # ML_X_Y_Z
    }
    
    for col in ml_cols:
        depth = col.count('_')
        if depth == 1:
            level_counts['level1'] += 1
        elif depth == 2:
            level_counts['level2'] += 1
        elif depth >= 3:
            level_counts['level3'] += 1
            
    return level_counts
