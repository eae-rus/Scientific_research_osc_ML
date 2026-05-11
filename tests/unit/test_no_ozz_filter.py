"""Тесты для модуля no_ozz_filter."""

import pytest
import polars as pl

from osc_tools.data_management.no_ozz_filter import (
    get_no_ozz_files,
    filter_no_ozz_dataframe,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def df_with_ozz() -> pl.DataFrame:
    """DataFrame с 3 файлами: 2 без ОЗЗ, 1 с ОЗЗ."""
    return pl.DataFrame({
        'file_name': ['file_a'] * 4 + ['file_b'] * 4 + ['file_c'] * 4,
        'IA': [0.1] * 12,
        'UA': [10.0] * 12,
        # file_a: нет ОЗЗ
        # file_b: есть ML_2_1 = 1 в последних 2 строках
        # file_c: нет ОЗЗ
        'ML_2_1':   [0, 0, 0, 0,   0, 0, 1, 1,   0, 0, 0, 0],
        'ML_2_1_1': [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
        'ML_2_1_2': [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
        'ML_2_1_3': [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
        'ML_2_2':   [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0],
    })


@pytest.fixture
def df_all_clean() -> pl.DataFrame:
    """DataFrame: все файлы без ОЗЗ."""
    return pl.DataFrame({
        'file_name': ['clean1'] * 3 + ['clean2'] * 3,
        'IA': [0.5] * 6,
        'ML_2_1': [0] * 6,
        'ML_2_1_1': [0] * 6,
        'ML_2_1_2': [0] * 6,
        'ML_2_1_3': [0] * 6,
        'ML_2_2': [0] * 6,
    })


@pytest.fixture
def df_string_columns() -> pl.DataFrame:
    """DataFrame с OZZ-колонками типа String (как бывает в CSV)."""
    return pl.DataFrame({
        'file_name': ['s1'] * 2 + ['s2'] * 2,
        'ML_2_1': ['0', '0', '1', '0'],
        'ML_2_1_1': ['0', '0', '0', '0'],
        'ML_2_1_2': ['0', '0', '0', '0'],
        'ML_2_1_3': ['0', '0', '0', '0'],
        'ML_2_2': ['0', '0', '0', '0'],
    })


# -----------------------------------------------------------------------
# get_no_ozz_files
# -----------------------------------------------------------------------

class TestGetNoOzzFiles:

    def test_basic_filtering(self, df_with_ozz: pl.DataFrame):
        """Файл с OZZ отфильтрован, два чистых остаются."""
        names, stats = get_no_ozz_files(df_with_ozz, verbose=False)
        assert sorted(names) == ['file_a', 'file_c']
        assert stats['with_ozz'] == 1
        assert stats['without_ozz'] == 2
        assert stats['total_files'] == 3

    def test_all_clean(self, df_all_clean: pl.DataFrame):
        """Все файлы чистые."""
        names, stats = get_no_ozz_files(df_all_clean, verbose=False)
        assert len(names) == 2
        assert stats['with_ozz'] == 0

    def test_string_columns_handled(self, df_string_columns: pl.DataFrame):
        """String-колонки корректно приводятся к числовым."""
        names, stats = get_no_ozz_files(df_string_columns, verbose=False)
        assert 's1' in names
        assert 's2' not in names  # s2 имеет ML_2_1=1

    def test_missing_ozz_columns_raises(self):
        """Ошибка если нет ни одной OZZ-колонки."""
        df = pl.DataFrame({'file_name': ['a'], 'IA': [1.0]})
        with pytest.raises(ValueError, match="нет ни одной OZZ"):
            get_no_ozz_files(df, verbose=False)

    def test_partial_ozz_columns(self):
        """Работает с подмножеством OZZ-колонок (не все 5)."""
        df = pl.DataFrame({
            'file_name': ['a'] * 2 + ['b'] * 2,
            'ML_2_1': [0, 0, 1, 0],
            # Нет ML_2_1_1, ML_2_1_2, ML_2_1_3, ML_2_2
        })
        names, stats = get_no_ozz_files(df, verbose=False)
        assert names == ['a']

    def test_verbose_output(self, df_with_ozz: pl.DataFrame, capsys):
        """verbose=True печатает статистику."""
        get_no_ozz_files(df_with_ozz, verbose=True)
        captured = capsys.readouterr()
        assert 'без ОЗЗ: 2' in captured.out


# -----------------------------------------------------------------------
# filter_no_ozz_dataframe
# -----------------------------------------------------------------------

class TestFilterNoOzzDataframe:

    def test_filter_with_precomputed(self, df_with_ozz: pl.DataFrame):
        """Фильтрация с предвычисленным списком файлов."""
        result = filter_no_ozz_dataframe(
            df_with_ozz, no_ozz_files=['file_a', 'file_c'], verbose=False,
        )
        assert result.height == 8  # 4 + 4
        assert set(result['file_name'].unique().to_list()) == {'file_a', 'file_c'}

    def test_filter_auto(self, df_with_ozz: pl.DataFrame):
        """Фильтрация с автоматическим определением файлов."""
        result = filter_no_ozz_dataframe(df_with_ozz, verbose=False)
        assert result.height == 8
        assert 'file_b' not in result['file_name'].to_list()

    def test_empty_result(self):
        """Все файлы с ОЗЗ → пустой результат."""
        df = pl.DataFrame({
            'file_name': ['bad'] * 3,
            'ML_2_1': [1, 1, 1],
            'ML_2_1_1': [0, 0, 0],
            'ML_2_1_2': [0, 0, 0],
            'ML_2_1_3': [0, 0, 0],
            'ML_2_2': [0, 0, 0],
        })
        result = filter_no_ozz_dataframe(df, verbose=False)
        assert result.height == 0
