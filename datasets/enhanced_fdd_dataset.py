from datasets.base_dataset import BaseDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from collections import Counter
from common.utils import get_short_names_ml_signals
from datasets.base_dataset import BaseDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans  # Добавьте эту строку
import pandas as pd
import numpy as np
from collections import Counter
from common.utils import get_short_names_ml_signals


class EnhancedFDDDataset(BaseDataset):
    """
    Улучшенная версия FDDDataset с продвинутыми стратегиями разбиения
    """

    def __init__(self, csv_path=None, split_strategy='original', **kwargs):
        """
        Args:
            csv_path: путь к CSV файлу
            split_strategy: стратегия разбиения данных
                - 'original': оригинальная стратегия
                - 'temporal_aware': временно-осведомленное разбиение
                - 'emergency_stratified': усиленная стратификация по авариям
                - 'clustering_based': разбиение на основе кластеризации
                - 'balanced_minority': сбалансированное разбиение редких классов
            **kwargs: дополнительные параметры для стратегий
        """
        self.split_strategy = split_strategy
        self.split_kwargs = kwargs
        super().__init__(csv_path)
        self.df = self.df[['IA', 'IC', 'UA BB', 'UB BB', 'UC BB']]

    def _set_target(self):
        """Создание целевой переменной (аналогично оригинальному коду)"""
        # Получаем списки сигналов для каждого события
        _, ml_opr_swch, ml_abnorm_evnt, ml_emerg_evnt = get_short_names_ml_signals()

        # Маски для каждого события
        target_opr = self.df.filter(ml_opr_swch).any(axis=1).astype(int)
        target_abnorm = self.df.filter(ml_abnorm_evnt).any(axis=1).astype(int)
        target_emerg = self.df.filter(ml_emerg_evnt).any(axis=1).astype(int)

        # Вычисляем нормальное состояние
        norm_state = ((target_opr + target_abnorm + target_emerg) == 0).astype(int)

        # Создаем target с приоритетом: emerg_evnt (2) > abnorm_evnt (1) > (opr_swch или norm_state) (0)
        conditions = [
            target_emerg.astype(bool),
            target_abnorm.astype(bool),
            (target_opr.astype(bool) | norm_state.astype(bool))
        ]
        choices = [2, 1, 0]
        target = pd.Series(np.select(conditions, choices, default=0), index=self.df.index, name='target')

        return target

    def _train_test_split(self):
        """Применяет выбранную стратегию разбиения"""

        if self.split_strategy == 'original':
            return self._original_split()
        elif self.split_strategy == 'temporal_aware':
            return self._temporal_aware_split()
        elif self.split_strategy == 'emergency_stratified':
            return self._emergency_stratified_split()
        elif self.split_strategy == 'clustering_based':
            return self._clustering_based_split()
        elif self.split_strategy == 'balanced_minority':
            return self._balanced_minority_split()
        else:
            print(f"⚠ Неизвестная стратегия: {self.split_strategy}, используется оригинальная")
            return self._original_split()

    def _original_split(self):
        """Оригинальная стратегия разбиения"""
        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()

        # Стратификация по аварийным событиям
        file_labels = self.target.groupby(file_groups).max()
        strat_col = (file_labels == 2).astype(int)

        # (train_val + test)
        files_train_val, files_test = train_test_split(
            unique_files,
            test_size=0.2,
            stratify=strat_col,
            random_state=42
        )

        # (train + val)
        strat_train_val = (file_labels.loc[files_train_val] == 2).astype(int)
        files_train, files_val = train_test_split(
            files_train_val,
            test_size=0.1,
            stratify=strat_train_val,
            random_state=42
        )

        return (
            file_groups.isin(files_train),
            file_groups.isin(files_val),
            file_groups.isin(files_test)
        )

    def _temporal_aware_split(self):
        """Временно-осведомленное разбиение"""
        print("Применяется временно-осведомленное разбиение...")

        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = sorted(file_groups.unique())  # Сортируем по времени

        test_size = self.split_kwargs.get('test_size', 0.2)
        val_size = self.split_kwargs.get('val_size', 0.1)

        n_files = len(unique_files)
        n_test = int(n_files * test_size)
        n_val = int(n_files * val_size)
        n_train = n_files - n_test - n_val

        # Старые данные для обучения, новые для тестирования
        files_train = unique_files[:n_train]
        files_val = unique_files[n_train:n_train + n_val]
        files_test = unique_files[n_train + n_val:]

        train_mask = file_groups.isin(files_train)
        val_mask = file_groups.isin(files_val)
        test_mask = file_groups.isin(files_test)

        self._print_split_stats(train_mask, val_mask, test_mask, "Temporal-Aware")

        return train_mask, val_mask, test_mask

    def _emergency_stratified_split(self):
        """Стратифицированное разбиение с усилением аварийных случаев"""
        print("Применяется разбиение с усилением аварийных случаев...")

        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()

        test_size = self.split_kwargs.get('test_size', 0.2)
        val_size = self.split_kwargs.get('val_size', 0.1)

        # Анализ файлов на наличие аварий
        file_labels = []
        for file in unique_files:
            file_mask = file_groups == file
            file_targets = self.target[file_mask]

            has_emergency = (file_targets == 2).any()
            emergency_ratio = (file_targets == 2).sum() / len(file_targets)

            if has_emergency:
                if emergency_ratio > 0.1:
                    label = 'high_emergency'
                else:
                    label = 'low_emergency'
            else:
                has_anomaly = (file_targets == 1).any()
                label = 'anomaly' if has_anomaly else 'normal'

            file_labels.append(label)

        # Проверяем распределение классов
        from collections import Counter
        label_counts = Counter(file_labels)
        print(f"   Распределение файлов: {dict(label_counts)}")

        # Если какой-то класс имеет только 1 файл, используем простое разбиение
        if min(label_counts.values()) < 2:
            print("   Обнаружены классы с единственным файлом, используем простое разбиение...")
            return self._simple_split()

        try:
            # Стратифицированное разбиение
            files_train_val, files_test = train_test_split(
                unique_files,
                test_size=test_size,
                stratify=file_labels,
                random_state=42
            )

            # Для train/val тоже проверяем возможность стратификации
            labels_train_val = [file_labels[i] for i, f in enumerate(unique_files) if f in files_train_val]
            label_counts_tv = Counter(labels_train_val)

            if min(label_counts_tv.values()) < 2:
                # Простое разбиение train/val
                files_train, files_val = train_test_split(
                    files_train_val,
                    test_size=val_size / (1 - test_size),
                    random_state=42
                )
            else:
                files_train, files_val = train_test_split(
                    files_train_val,
                    test_size=val_size / (1 - test_size),
                    stratify=labels_train_val,
                    random_state=42
                )

        except ValueError as e:
            print(f"   Ошибка стратификации: {e}")
            print("   Переключаемся на простое разбиение...")
            return self._simple_split()

        train_mask = file_groups.isin(files_train)
        val_mask = file_groups.isin(files_val)
        test_mask = file_groups.isin(files_test)

        self._print_split_stats(train_mask, val_mask, test_mask, "Emergency-Stratified")

        return train_mask, val_mask, test_mask

    def _simple_split(self):
        """Простое разбиение без стратификации, но с гарантией аварий во всех выборках"""
        print("Применяется простое разбиение с контролем аварий...")

        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()

        test_size = self.split_kwargs.get('test_size', 0.2)
        val_size = self.split_kwargs.get('val_size', 0.1)

        # Разделяем файлы на те, что содержат аварии, и остальные
        emergency_files = []
        normal_files = []

        for file in unique_files:
            file_mask = file_groups == file
            file_targets = self.target[file_mask]

            if (file_targets == 2).any():
                emergency_files.append(file)
            else:
                normal_files.append(file)

        print(f"   Файлов с авариями: {len(emergency_files)}")
        print(f"   Файлов без аварий: {len(normal_files)}")

        # Перемешиваем файлы
        import random
        seed = self.split_kwargs.get('random_state', 42)  # ← ИСПОЛЬЗУЕМ ПЕРЕДАННЫЙ SEED
        random.seed(seed)
        random.shuffle(emergency_files)
        random.shuffle(normal_files)
        print(f"   Используется random seed: {seed}")

        # Распределяем файлы с авариями равномерно
        n_emergency = len(emergency_files)
        if n_emergency >= 3:  # Если есть минимум 3 файла с авариями
            n_emergency_test = max(1, int(n_emergency * test_size))
            n_emergency_val = max(1, int(n_emergency * val_size))
            n_emergency_train = n_emergency - n_emergency_test - n_emergency_val

            emergency_train = emergency_files[:n_emergency_train]
            emergency_val = emergency_files[n_emergency_train:n_emergency_train + n_emergency_val]
            emergency_test = emergency_files[n_emergency_train + n_emergency_val:]
        else:
            # Если файлов с авариями мало, распределяем как можем
            emergency_train = emergency_files[:max(1, len(emergency_files) - 2)]
            emergency_val = emergency_files[len(emergency_train):len(emergency_train) + 1] if len(
                emergency_files) > 1 else []
            emergency_test = emergency_files[len(emergency_train) + len(emergency_val):]

        # Распределяем обычные файлы
        n_normal = len(normal_files)
        n_normal_test = int(n_normal * test_size)
        n_normal_val = int(n_normal * val_size)

        normal_train = normal_files[:n_normal - n_normal_test - n_normal_val]
        normal_val = normal_files[n_normal - n_normal_test - n_normal_val:n_normal - n_normal_test]
        normal_test = normal_files[n_normal - n_normal_test:]

        # Объединяем
        files_train = emergency_train + normal_train
        files_val = emergency_val + normal_val
        files_test = emergency_test + normal_test

        train_mask = file_groups.isin(files_train)
        val_mask = file_groups.isin(files_val)
        test_mask = file_groups.isin(files_test)

        self._print_split_stats(train_mask, val_mask, test_mask, "Simple-Split")

        return train_mask, val_mask, test_mask

    def _clustering_based_split(self):
        """Разбиение на основе кластеризации"""
        print("Применяется разбиение на основе кластеризации...")

        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()

        test_size = self.split_kwargs.get('test_size', 0.2)
        val_size = self.split_kwargs.get('val_size', 0.1)
        n_clusters = self.split_kwargs.get('n_clusters',
                                           min(10, len(unique_files) // 3))  # Адаптивное количество кластеров

        # Вычисляем характеристики файлов
        file_features = []
        for file in unique_files:
            file_mask = file_groups == file
            file_data = self.df[file_mask]
            file_targets = self.target[file_mask]

            features = [
                file_data.mean().mean(),  # Средние значения
                file_data.std().mean(),  # Стандартные отклонения
                file_data.var().mean(),  # Дисперсия
                (file_targets == 0).mean(),  # Доля нормальных
                (file_targets == 1).mean(),  # Доля аномалий
                (file_targets == 2).mean(),  # Доля аварий
                len(file_data),  # Длина файла
                file_data.max().max(),  # Максимальные значения
                file_data.min().min(),  # Минимальные значения
            ]
            file_features.append(features)

        # Кластеризация
        from sklearn.cluster import KMeans
        seed = self.split_kwargs.get('random_state', 42)  # Используем переданный seed
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(file_features)

        # Проверяем распределение кластеров
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)

        # Если есть кластеры с одним файлом, объединяем их с ближайшими
        if np.min(cluster_counts) < 2:
            print(f"   Обнаружены малые кластеры, применяем упрощенную стратегию...")
            # Используем простое случайное разбиение с учетом аварий
            return self._emergency_stratified_split()

        try:
            # Стратифицированное разбиение по кластерам
            files_train_val, files_test = train_test_split(
                unique_files,
                test_size=test_size,
                stratify=cluster_labels,
                random_state=42
            )

            # Получаем кластеры для train_val
            train_val_indices = [i for i, f in enumerate(unique_files) if f in files_train_val]
            train_val_clusters = cluster_labels[train_val_indices]

            # Проверяем, можно ли разбить train_val
            unique_tv_clusters, tv_counts = np.unique(train_val_clusters, return_counts=True)
            if np.min(tv_counts) < 2:
                # Простое разбиение без стратификации
                files_train, files_val = train_test_split(
                    files_train_val,
                    test_size=val_size / (1 - test_size),
                    random_state=42
                )
            else:
                files_train, files_val = train_test_split(
                    files_train_val,
                    test_size=val_size / (1 - test_size),
                    stratify=train_val_clusters,
                    random_state=42
                )

        except ValueError as e:
            print(f"   Ошибка стратификации: {e}")
            print("   Переключаемся на emergency_stratified стратегию...")
            return self._emergency_stratified_split()

        train_mask = file_groups.isin(files_train)
        val_mask = file_groups.isin(files_val)
        test_mask = file_groups.isin(files_test)

        self._print_split_stats(train_mask, val_mask, test_mask, "Clustering-Based")

        return train_mask, val_mask, test_mask

    def _balanced_minority_split(self):
        """Сбалансированное разбиение для редких классов"""
        print(" Применяется сбалансированное разбиение для редких классов...")

        file_groups = self.df.index.get_level_values('file_name').str[:32]
        unique_files = file_groups.unique()

        test_size = self.split_kwargs.get('test_size', 0.2)
        val_size = self.split_kwargs.get('val_size', 0.1)

        # Классификация файлов
        file_info = []
        for file in unique_files:
            file_mask = file_groups == file
            file_targets = self.target[file_mask]

            class_counts = [(file_targets == i).sum() for i in range(3)]
            emergency_count = class_counts[2]

            file_info.append({
                'file': file,
                'emergency_count': emergency_count,
                'total_count': len(file_targets),
                'emergency_ratio': emergency_count / len(file_targets) if len(file_targets) > 0 else 0
            })

        # Разделение на файлы с авариями и без
        emergency_files = [f for f in file_info if f['emergency_count'] > 0]
        normal_files = [f for f in file_info if f['emergency_count'] == 0]

        # Равномерное распределение файлов с авариями
        np.random.seed(42)
        np.random.shuffle(emergency_files)
        np.random.shuffle(normal_files)

        # Распределение аварийных файлов
        n_emergency = len(emergency_files)
        if n_emergency > 0:
            n_emergency_test = max(1, int(n_emergency * test_size))
            n_emergency_val = max(1, int(n_emergency * val_size))
            n_emergency_train = n_emergency - n_emergency_test - n_emergency_val

            emergency_train = emergency_files[:n_emergency_train]
            emergency_val = emergency_files[n_emergency_train:n_emergency_train + n_emergency_val]
            emergency_test = emergency_files[n_emergency_train + n_emergency_val:]
        else:
            emergency_train = emergency_val = emergency_test = []

        # Распределение обычных файлов
        n_normal = len(normal_files)
        n_normal_test = int(n_normal * test_size)
        n_normal_val = int(n_normal * val_size)

        normal_train = normal_files[:n_normal - n_normal_test - n_normal_val]
        normal_val = normal_files[n_normal - n_normal_test - n_normal_val:n_normal - n_normal_test]
        normal_test = normal_files[n_normal - n_normal_test:]

        # Объединение
        files_train = [f['file'] for f in emergency_train + normal_train]
        files_val = [f['file'] for f in emergency_val + normal_val]
        files_test = [f['file'] for f in emergency_test + normal_test]

        train_mask = file_groups.isin(files_train)
        val_mask = file_groups.isin(files_val)
        test_mask = file_groups.isin(files_test)

        self._print_split_stats(train_mask, val_mask, test_mask, "Balanced-Minority")

        return train_mask, val_mask, test_mask

    def _print_split_stats(self, train_mask, val_mask, test_mask, strategy_name):
        """Печатает статистику распределения классов"""

        print(f"\n Статистика для стратегии: {strategy_name}")
        print("-" * 70)

        for split_name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
            split_targets = self.target[mask]
            total = len(split_targets)

            if total == 0:
                continue

            class_counts = [(split_targets == i).sum() for i in range(3)]
            class_percentages = [count / total * 100 for count in class_counts]

            print(f"{split_name:>5}: {total:>7} samples | "
                  f"Normal: {class_counts[0]:>6} ({class_percentages[0]:>5.1f}%) | "
                  f"Anomaly: {class_counts[1]:>6} ({class_percentages[1]:>5.1f}%) | "
                  f"Emergency: {class_counts[2]:>6} ({class_percentages[2]:>5.1f}%)")

        print("-" * 70)

        # Дополнительная статистика для аварий
        emergency_train = (self.target[train_mask] == 2).sum()
        emergency_val = (self.target[val_mask] == 2).sum()
        emergency_test = (self.target[test_mask] == 2).sum()
        total_emergency = emergency_train + emergency_val + emergency_test

        if total_emergency > 0:
            print(
                f" Распределение аварий: Train: {emergency_train} ({emergency_train / total_emergency * 100:.1f}%) | "
                f"Val: {emergency_val} ({emergency_val / total_emergency * 100:.1f}%) | "
                f"Test: {emergency_test} ({emergency_test / total_emergency * 100:.1f}%)")
            print("-" * 70)

    def get_split_info(self):
        """Возвращает информацию о разбиении"""
        return {
            'strategy': self.split_strategy,
            'train_samples': self.train_mask.sum(),
            'val_samples': self.val_mask.sum(),
            'test_samples': self.test_mask.sum(),
            'train_emergency': (self.target[self.train_mask] == 2).sum(),
            'val_emergency': (self.target[self.val_mask] == 2).sum(),
            'test_emergency': (self.target[self.test_mask] == 2).sum(),
        }