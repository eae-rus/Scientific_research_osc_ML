import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Добавляем корень проекта в путь импорта
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from osc_tools.ml.models.kan import ConvKAN
from osc_tools.visualization.kan_plot import plot_kan_layer_grid

def main():
    # Путь к модели
    model_path = ROOT_DIR / 'experiments' / 'advanced_phase2' / 'ConvKAN_Matched.pt'
    
    if not model_path.exists():
        print(f"Файл модели не найден: {model_path}")
        return

    print(f"Загрузка модели из {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Пытаемся определить num_classes из весов последнего слоя
    # Структура ConvKAN: classifier[1] - это последний KANLinear
    # Ключи в state_dict будут выглядеть как classifier.1.base_weight
    
    if 'classifier.1.base_weight' in state_dict:
        out_features, in_features = state_dict['classifier.1.base_weight'].shape
        num_classes = out_features
        print(f"Определено количество классов: {num_classes}")
    else:
        print("Не удалось определить количество классов автоматически. Используем значение по умолчанию (10).")
        num_classes = 10

    # Параметры модели из run_phase2_advanced.py
    # kan_params = {
    #     "in_channels": 12,
    #     "num_classes": num_classes,
    #     "grid_size": 10,
    #     "base_filters": 8
    # }
    
    model = ConvKAN(
        in_channels=12,
        num_classes=num_classes,
        grid_size=10,
        base_filters=8
    )
    
    try:
        model.load_state_dict(state_dict)
        print("Веса успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        return

    # Визуализация
    print("\n--- Визуализация первого сверточного слоя (features[0]) ---")
    # features[0] is KANConv1d. It has a .kan_layer attribute which is KANLinear
    if hasattr(model.features[0], 'kan_layer'):
        conv_layer = model.features[0].kan_layer
        print(f"Визуализация KANLinear внутри Conv1d (входы: {conv_layer.in_features}, выходы: {conv_layer.out_features})")
        plot_kan_layer_grid(conv_layer, max_inputs=5, max_outputs=5)
    else:
        print("features[0] не имеет атрибута kan_layer.")

    print("\n--- Визуализация первого слоя классификатора (classifier[0]) ---")
    layer = model.classifier[0] # KANLinear(base_filters*4, base_filters*2)
    # base_filters=8 -> 32 inputs, 16 outputs.
    # Let's plot a subset.
    
    print("Построение графиков...")
    plot_kan_layer_grid(layer, max_inputs=5, max_outputs=5)
    print("Готово. График должен открыться в окне (или сохраниться, если настроен backend).")

if __name__ == "__main__":
    main()
