"""Тесты для NeutralChannelDropout и CompositeAugmenter."""

import numpy as np
import pytest
import torch

from osc_tools.ml.augmentation import (
    CompositeAugmenter,
    NeutralChannelDropout,
    TimeSeriesAugmenter,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def raw_2d() -> np.ndarray:
    """Сырые данные (T, 8) с уникальными значениями по каналам."""
    T = 50
    data = np.zeros((T, 8), dtype=np.float32)
    for ch in range(8):
        data[:, ch] = (ch + 1) * 10.0  # 10, 20, 30, 40, 50, 60, 70, 80
    return data


@pytest.fixture
def raw_3d() -> torch.Tensor:
    """Сырые данные (B, T, 8) — torch."""
    B, T = 4, 50
    data = torch.zeros(B, T, 8)
    for ch in range(8):
        data[:, :, ch] = (ch + 1) * 10.0
    return data


# -----------------------------------------------------------------------
# NeutralChannelDropout — 2D numpy
# -----------------------------------------------------------------------

class TestNeutralChannelDropout2D:
    """Тесты на numpy (T, 8) входах."""

    def test_all_mode_keeps_data(self, raw_2d: np.ndarray):
        """Режим 'all' (p_all=1) не меняет данные."""
        aug = NeutralChannelDropout(fill_value=0.0, p_all=1.0, p_drop_in=0.0, p_drop_in_un=0.0)
        out = aug(raw_2d)
        np.testing.assert_array_equal(out, raw_2d)

    def test_drop_in_mode(self, raw_2d: np.ndarray):
        """Режим drop_in (p_drop_in=1) обнуляет только IN (индекс 3)."""
        aug = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0)
        out = aug(raw_2d)
        assert out[:, 3].sum() == 0.0, "IN (idx=3) должен быть = 0"
        # Остальные каналы не тронуты
        for ch in [0, 1, 2, 4, 5, 6, 7]:
            np.testing.assert_array_equal(out[:, ch], raw_2d[:, ch])

    def test_drop_in_un_mode(self, raw_2d: np.ndarray):
        """Режим drop_in_un (p_drop_in_un=1) обнуляет IN и UN."""
        aug = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=0.0, p_drop_in_un=1.0)
        out = aug(raw_2d)
        assert out[:, 3].sum() == 0.0, "IN (idx=3) должен быть = 0"
        assert out[:, 7].sum() == 0.0, "UN (idx=7) должен быть = 0"
        for ch in [0, 1, 2, 4, 5, 6]:
            np.testing.assert_array_equal(out[:, ch], raw_2d[:, ch])

    def test_custom_fill_value(self, raw_2d: np.ndarray):
        """fill_value корректно применяется."""
        aug = NeutralChannelDropout(fill_value=-999.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0)
        out = aug(raw_2d)
        np.testing.assert_array_equal(out[:, 3], np.full(50, -999.0))

    def test_does_not_modify_original(self, raw_2d: np.ndarray):
        """Оригинальный массив не модифицируется (копия)."""
        original = raw_2d.copy()
        aug = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=0.0, p_drop_in_un=1.0)
        _ = aug(raw_2d)
        np.testing.assert_array_equal(raw_2d, original)

    def test_output_shape_preserved(self, raw_2d: np.ndarray):
        """Форма выхода совпадает с входом."""
        aug = NeutralChannelDropout()
        out = aug(raw_2d)
        assert out.shape == raw_2d.shape


# -----------------------------------------------------------------------
# NeutralChannelDropout — 3D torch
# -----------------------------------------------------------------------

class TestNeutralChannelDropout3D:
    """Тесты на torch (B, T, 8) входах."""

    def test_drop_in_torch(self, raw_3d: torch.Tensor):
        """drop_in работает с torch тензорами."""
        aug = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0)
        out = aug(raw_3d)
        assert isinstance(out, torch.Tensor)
        assert out[:, :, 3].sum().item() == 0.0
        # IA не тронута
        torch.testing.assert_close(out[:, :, 0], raw_3d[:, :, 0])

    def test_drop_in_un_torch(self, raw_3d: torch.Tensor):
        """drop_in_un работает с torch тензорами."""
        aug = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=0.0, p_drop_in_un=1.0)
        out = aug(raw_3d)
        assert out[:, :, 3].sum().item() == 0.0
        assert out[:, :, 7].sum().item() == 0.0

    def test_does_not_modify_original_torch(self, raw_3d: torch.Tensor):
        """Оригинальный тензор не модифицируется."""
        original = raw_3d.clone()
        aug = NeutralChannelDropout(p_all=0.0, p_drop_in=0.0, p_drop_in_un=1.0)
        _ = aug(raw_3d)
        torch.testing.assert_close(raw_3d, original)


# -----------------------------------------------------------------------
# NeutralChannelDropout — вероятности
# -----------------------------------------------------------------------

class TestNeutralChannelDropoutProbabilities:
    """Статистические тесты на распределение режимов."""

    def test_probabilities_sum_to_one(self):
        """Внутренние вероятности нормализуются."""
        aug = NeutralChannelDropout(p_all=2.0, p_drop_in=3.0, p_drop_in_un=5.0)
        assert abs(aug.p_all - 0.2) < 1e-6
        assert abs(aug.p_drop_in - 0.3) < 1e-6

    def test_statistical_distribution(self, raw_2d: np.ndarray):
        """Примерно 1/3 каждого режима при равных вероятностях."""
        aug = NeutralChannelDropout(fill_value=0.0)
        n_all = n_drop_in = n_drop_in_un = 0
        N = 3000
        for _ in range(N):
            out = aug(raw_2d)
            in_zero = out[:, 3].sum() == 0.0
            un_zero = out[:, 7].sum() == 0.0
            if not in_zero and not un_zero:
                n_all += 1
            elif in_zero and not un_zero:
                n_drop_in += 1
            else:
                n_drop_in_un += 1

        # Каждый режим ~1/3, допуск ±5%
        for count, name in [(n_all, 'all'), (n_drop_in, 'drop_in'), (n_drop_in_un, 'drop_in_un')]:
            ratio = count / N
            assert 0.25 < ratio < 0.42, f"Режим {name}: {ratio:.2f} (ожидалось ~0.33)"


# -----------------------------------------------------------------------
# CompositeAugmenter
# -----------------------------------------------------------------------

class TestCompositeAugmenter:

    def test_empty(self, raw_2d: np.ndarray):
        """Пустой комбайн не меняет данные."""
        aug = CompositeAugmenter()
        out = aug(raw_2d)
        np.testing.assert_array_equal(out, raw_2d)

    def test_single(self, raw_2d: np.ndarray):
        """С одним аугментатором — эквивалентен ему."""
        dropout = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0)
        aug = CompositeAugmenter(dropout)
        out = aug(raw_2d)
        assert out[:, 3].sum() == 0.0

    def test_none_ignored(self, raw_2d: np.ndarray):
        """None-аугментаторы пропускаются."""
        dropout = NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0)
        aug = CompositeAugmenter(None, dropout, None)
        out = aug(raw_2d)
        assert out[:, 3].sum() == 0.0

    def test_sequential_application(self, raw_2d: np.ndarray):
        """Два аугментатора применяются последовательно."""
        # Первый инвертирует (всё * -1), второй обнуляет IN
        class Inverter:
            def __call__(self, x):
                return -x

        aug = CompositeAugmenter(
            Inverter(),
            NeutralChannelDropout(fill_value=0.0, p_all=0.0, p_drop_in=1.0, p_drop_in_un=0.0),
        )
        out = aug(raw_2d)
        # IN обнулён (после инверсии второй аугментатор зануляет)
        assert out[:, 3].sum() == 0.0
        # IA = -10 (инвертировано)
        np.testing.assert_array_equal(out[:, 0], -raw_2d[:, 0])

    def test_output_shape(self, raw_2d: np.ndarray):
        """Форма не меняется."""
        aug = CompositeAugmenter(NeutralChannelDropout())
        assert aug(raw_2d).shape == raw_2d.shape
