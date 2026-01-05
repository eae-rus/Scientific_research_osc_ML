import polars as pl
from typing import List, Dict

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

def get_target_columns(level: str = 'base') -> List[str]:
    """
    Возвращает список целевых колонок для указанного уровня.

    Args:
        level: 'base' (4 класса) или 'full' (все колонки ML_)
    """
    if level == 'base':
        return ["Target_Normal", "Target_ML_1", "Target_ML_2", "Target_ML_3"]
    elif level == 'full':
        # Для уровня 'full' требуется знать список колонок в DataFrame.
        # Обычно нужно вызывать `get_ml_columns(df)` с самим датафреймом.
        raise ValueError("Для уровня 'full' используйте get_ml_columns(df)")
    else:
        raise ValueError(f"Unknown level: {level}")
