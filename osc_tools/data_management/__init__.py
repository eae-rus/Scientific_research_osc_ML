"""Инструменты данных; тяжёлый менеджер датасетов загружается только по запросу."""
__all__ = ['DatasetManager', 'initialize_datasets']


def __getattr__(name):
    """Сохранить прежние импорты без зависимостей менеджера при простом поиске."""
    if name in {'DatasetManager', 'initialize_datasets'}:
        from .dataset_manager import DatasetManager, initialize_datasets
        return {'DatasetManager':DatasetManager,'initialize_datasets':initialize_datasets}[name]
    raise AttributeError(name)
