"""Тесты для BalancedConcatDataset и BalancedEpochSampler."""

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from osc_tools.ml.balanced_dataset import BalancedConcatDataset, BalancedEpochSampler


# -----------------------------------------------------------------------
# Вспомогательный датасет
# -----------------------------------------------------------------------

class DummyDataset(Dataset):
    """Простой датасет для тестов: возвращает (Tensor(idx), Tensor(0))."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return torch.tensor([float(idx)]), torch.tensor([0.0])


# -----------------------------------------------------------------------
# BalancedConcatDataset
# -----------------------------------------------------------------------

class TestBalancedConcatDataset:

    def test_basic_concat(self):
        """Два датасета объединяются корректно."""
        ds1 = DummyDataset(10)
        ds2 = DummyDataset(5)
        class_ids = [0] * 10 + [1] * 5
        bcd = BalancedConcatDataset([ds1, ds2], class_ids)
        assert len(bcd) == 15

    def test_class_indices(self):
        """indices_by_class содержит правильные индексы."""
        ds1 = DummyDataset(3)
        ds2 = DummyDataset(2)
        class_ids = [0, 0, 0, 1, 1]
        bcd = BalancedConcatDataset([ds1, ds2], class_ids)
        np.testing.assert_array_equal(bcd.indices_by_class[0], [0, 1, 2])
        np.testing.assert_array_equal(bcd.indices_by_class[1], [3, 4])

    def test_getitem_first_dataset(self):
        """Элемент из первого датасета возвращается корректно."""
        ds1 = DummyDataset(5)
        ds2 = DummyDataset(3)
        class_ids = [0] * 5 + [1] * 3
        bcd = BalancedConcatDataset([ds1, ds2], class_ids)
        X, Y = bcd[2]
        assert X.item() == 2.0

    def test_getitem_second_dataset(self):
        """Элемент из второго датасета: индексация корректна."""
        ds1 = DummyDataset(5)
        ds2 = DummyDataset(3)
        class_ids = [0] * 5 + [1] * 3
        bcd = BalancedConcatDataset([ds1, ds2], class_ids)
        X, Y = bcd[6]  # Индекс 6 -> ds2[1]
        assert X.item() == 1.0

    def test_class_ids_length_mismatch_raises(self):
        """Ошибка при несовпадении длин."""
        ds1 = DummyDataset(5)
        with pytest.raises(ValueError, match="class_ids length"):
            BalancedConcatDataset([ds1], class_ids=[0, 0])

    def test_multiple_classes(self):
        """5 классов создаются корректно."""
        datasets = [DummyDataset(10) for _ in range(5)]
        class_ids = []
        for cls_id in range(5):
            class_ids.extend([cls_id] * 10)
        bcd = BalancedConcatDataset(datasets, class_ids)
        assert len(bcd) == 50
        assert len(bcd.indices_by_class) == 5
        for c in range(5):
            assert len(bcd.indices_by_class[c]) == 10

    def test_print_stats(self, capsys):
        """print_stats не падает и выводит информацию."""
        ds = DummyDataset(3)
        bcd = BalancedConcatDataset([ds], class_ids=[0, 1, 0], class_names=['A', 'B'])
        bcd.print_stats()
        captured = capsys.readouterr()
        assert 'Всего элементов' in captured.out


# -----------------------------------------------------------------------
# BalancedEpochSampler
# -----------------------------------------------------------------------

class TestBalancedEpochSampler:

    @pytest.fixture
    def balanced_dataset(self) -> BalancedConcatDataset:
        """Датасет: класс 0 = 100 эл., класс 1 = 20 эл., класс 2 = 50 эл."""
        ds = DummyDataset(170)
        class_ids = [0] * 100 + [1] * 20 + [2] * 50
        return BalancedConcatDataset(
            [ds], class_ids, class_names=['big', 'small', 'medium'],
        )

    def test_length(self, balanced_dataset: BalancedConcatDataset):
        """Длина = n_classes * samples_per_class."""
        sampler = BalancedEpochSampler(balanced_dataset, seed=0)
        # samples_per_class = min(100, 20, 50) = 20
        assert len(sampler) == 3 * 20

    def test_custom_samples_per_class(self, balanced_dataset: BalancedConcatDataset):
        """Можно задать samples_per_class вручную."""
        sampler = BalancedEpochSampler(balanced_dataset, samples_per_class=10, seed=0)
        assert len(sampler) == 3 * 10

    def test_all_classes_represented(self, balanced_dataset: BalancedConcatDataset):
        """Каждый класс представлен в выборке."""
        sampler = BalancedEpochSampler(balanced_dataset, samples_per_class=15, seed=0)
        indices = list(sampler)
        class_ids = balanced_dataset.class_ids
        classes_seen = set(class_ids[i] for i in indices)
        assert classes_seen == {0, 1, 2}

    def test_balanced_counts(self, balanced_dataset: BalancedConcatDataset):
        """Каждый класс имеет одинаковое представительство."""
        sampler = BalancedEpochSampler(balanced_dataset, samples_per_class=15, seed=0)
        indices = list(sampler)
        class_ids = balanced_dataset.class_ids
        from collections import Counter
        counts = Counter(class_ids[i] for i in indices)
        assert counts[0] == 15
        assert counts[1] == 15
        assert counts[2] == 15

    def test_set_epoch_changes_order(self, balanced_dataset: BalancedConcatDataset):
        """set_epoch меняет порядок выборки."""
        sampler = BalancedEpochSampler(balanced_dataset, samples_per_class=10, seed=42)
        sampler.set_epoch(0)
        idx_ep0 = list(sampler)
        sampler.set_epoch(1)
        idx_ep1 = list(sampler)
        # С разными seed порядок скорее всего отличается
        assert idx_ep0 != idx_ep1

    def test_oversampling_small_class(self):
        """Малый класс oversampled c повтором."""
        ds = DummyDataset(105)
        class_ids = [0] * 100 + [1] * 5
        bcd = BalancedConcatDataset([ds], class_ids)
        sampler = BalancedEpochSampler(bcd, samples_per_class=50, seed=0)
        indices = list(sampler)
        class_ids_arr = bcd.class_ids
        c1_count = sum(1 for i in indices if class_ids_arr[i] == 1)
        assert c1_count == 50  # Oversampled

    def test_reproducibility(self, balanced_dataset: BalancedConcatDataset):
        """Одинаковые seed + epoch = одинаковая выборка."""
        s1 = BalancedEpochSampler(balanced_dataset, samples_per_class=10, seed=42)
        s2 = BalancedEpochSampler(balanced_dataset, samples_per_class=10, seed=42)
        s1.set_epoch(5)
        s2.set_epoch(5)
        assert list(s1) == list(s2)

    def test_indices_in_range(self, balanced_dataset: BalancedConcatDataset):
        """Все возвращаемые индексы в пределах [0, len(dataset))."""
        sampler = BalancedEpochSampler(balanced_dataset, samples_per_class=20, seed=0)
        indices = list(sampler)
        assert all(0 <= i < len(balanced_dataset) for i in indices)
