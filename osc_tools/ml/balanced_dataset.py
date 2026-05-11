"""
Объединённый датасет с балансировкой по классам.

Комбинирует 5 источников данных (4 типа ОЗЗ из SimOZZ + 1 класс «не ОЗЗ»
из реальных осциллограмм) и обеспечивает равномерную выборку из каждого
класса в каждую эпоху.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class BalancedConcatDataset(Dataset):
    """Объединяет несколько датасетов с одинаковым форматом (X, Y).

    Каждому исходному датасету или подмножеству присваивается *class_id*.
    ``BalancedEpochSampler`` использует ``class_id`` для равномерной выборки.

    При ``__getitem__`` возвращаемый Y дополняется колонкой класса источника
    **только если** ``inject_source_label=True`` (по умолчанию выключено —
    модель обучается тем же 4-колоночным Y).
    """

    def __init__(
        self,
        datasets: List[Dataset],
        class_ids: List[int],
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            datasets: список Dataset-ов. Все должны возвращать (X, Y)
                одинаковой формы.
            class_ids: для каждого *элемента* (суммарно) — id класса
                источника (0..4). Длина = сумма len(ds) по всем datasets.
            class_names: опционально, для логов.
        """
        super().__init__()
        self.datasets = datasets
        self._lengths = [len(ds) for ds in datasets]
        self._cumlen = np.cumsum(self._lengths)
        self._total = int(self._cumlen[-1])

        if len(class_ids) != self._total:
            raise ValueError(
                f"class_ids length {len(class_ids)} != total {self._total}"
            )
        self.class_ids = np.array(class_ids, dtype=np.int32)
        self.class_names = class_names or [str(i) for i in range(max(class_ids) + 1)]

        # Индексы по классам (для BalancedEpochSampler)
        self.indices_by_class: Dict[int, np.ndarray] = {}
        unique_classes = np.unique(self.class_ids)
        for c in unique_classes:
            self.indices_by_class[int(c)] = np.where(self.class_ids == c)[0]

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Находим, из какого датасета
        ds_idx = int(np.searchsorted(self._cumlen, idx, side='right'))
        local_idx = idx if ds_idx == 0 else idx - int(self._cumlen[ds_idx - 1])
        return self.datasets[ds_idx][local_idx]

    def print_stats(self) -> None:
        """Печатает распределение по классам."""
        print(f"[BalancedConcatDataset] Всего элементов: {self._total:,}")
        for c, idxs in sorted(self.indices_by_class.items()):
            name = self.class_names[c] if c < len(self.class_names) else str(c)
            print(f"  Класс {c} ({name}): {len(idxs):,} элементов")


class BalancedEpochSampler(Sampler):
    """Равномерная выборка из каждого класса за одну эпоху.

    За одну «эпоху» выбирает ``samples_per_class`` элементов из каждого
    класса. Если класс мельче — oversample с повтором.

    Args:
        dataset: ``BalancedConcatDataset``
        samples_per_class: число элементов из каждого класса за эпоху.
            None -> минимум среди всех классов.
        seed: random seed (для воспроизводимости; обновляется каждую эпоху).
    """

    def __init__(
        self,
        dataset: BalancedConcatDataset,
        samples_per_class: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self._epoch = 0

        if samples_per_class is None:
            # Минимальный класс определяет размер «эпохи»
            samples_per_class = min(
                len(idxs) for idxs in dataset.indices_by_class.values()
            )
        self.samples_per_class = samples_per_class
        self._n_classes = len(dataset.indices_by_class)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        indices = []
        for c, class_idxs in self.dataset.indices_by_class.items():
            n = len(class_idxs)
            if n >= self.samples_per_class:
                chosen = rng.choice(class_idxs, size=self.samples_per_class,
                                    replace=False)
            else:
                # Oversample с повтором
                chosen = rng.choice(class_idxs, size=self.samples_per_class,
                                    replace=True)
            indices.append(chosen)

        all_indices = np.concatenate(indices)
        rng.shuffle(all_indices)
        return iter(all_indices.tolist())

    def __len__(self) -> int:
        return self.samples_per_class * self._n_classes

    def set_epoch(self, epoch: int) -> None:
        """Обновляет random seed для очередной эпохи."""
        self._epoch = epoch
