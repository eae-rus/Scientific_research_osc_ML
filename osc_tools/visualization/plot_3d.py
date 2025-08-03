import os
import sys
import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt # Не используется в этой функции
# from mpl_toolkits.mplot3d import Axes3D # Не используется в этой функции
import plotly.graph_objects as go
import plotly.io as pio

# Добавляем ROOT_DIR, если скрипт не в корне проекта
try:
    # Попытка определить __file__ (работает при запуске как скрипт)
    current_file_dir = os.path.dirname(__file__)
except NameError:
    # __file__ не определен (например, в интерактивной сессии)
    # Используем текущую рабочую директорию как запасной вариант
    current_file_dir = os.getcwd()
    print(f"Warning: __file__ not defined. Using current working directory: {current_file_dir}")

ROOT_DIR = os.path.abspath(os.path.join(current_file_dir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    print(f"Added to sys.path: {ROOT_DIR}")

try:
    from skimage.measure import marching_cubes
except ImportError:
    print("Пожалуйста, установите scikit-image: pip install scikit-image")
    sys.exit() # Используем sys.exit() вместо exit() для большей совместимости

# --- Параметры нормализации (из вашего описания) ---
# Для модуля тока (I_pos_seq_mag)
I_mag_mean = 0.004592851620721061
I_mag_std = 0.15447972567465076

# Для модуля напряжения (V_pos_seq_mag)
V_mag_mean = 0.06707311131654074
V_mag_std = 1.034901411026839

# Если угол нормализовался как-то иначе, нужно это учесть здесь
I_angle_mean = 0.0 # Пример, если бы была нормализация
I_angle_std = np.pi  # Пример, если бы была нормализация (или другое значение)

# --- Функция для нормализации входных данных ---
def normalize_features(i_mag, i_angle, v_mag):
    """Нормализует признаки для подачи в модель."""
    norm_i_mag = (i_mag - I_mag_mean) / I_mag_std
    norm_v_mag = (v_mag - V_mag_mean) / V_mag_std
    # Нормализация угла (если она нужна)
    norm_i_angle = (i_angle - I_angle_mean) / I_angle_std # Используем плейсхолдеры
    # Если угол не нормализовался, просто верните i_angle
    # norm_i_angle = i_angle
    return norm_i_mag, norm_i_angle, norm_v_mag

# --- ФУНКЦИЯ ПРЕОБРАЗОВАНИЯ КООРДИНАТ ---
def polar_to_cartesian_current(i_mag, i_angle):
    """Преобразует модуль и угол тока в Re и Im."""
    re = i_mag * np.cos(i_angle)
    im = i_mag * np.sin(i_angle)
    return re, im

def cartesian_to_polar_current(re, im):
    """Преобразует Re и Im тока в модуль и угол."""
    i_mag = np.sqrt(re**2 + im**2)
    i_angle = np.arctan2(im, re)
    return i_mag, i_angle

def plot_3d_boundary_unified(
    model_path,
    device,
    output_html_file="decision_boundary_unified_3d.html",
    visualization_mode='boundary_and_fill', # Новый параметр: 'boundary_only', 'boundary_and_fill', 'heatmap'
    grid_density=50,       # Плотность сетки
    marker_size=2.5,       # Размер маркеров для 'boundary_and_fill' и 'heatmap'
    heatmap_opacity=0.6,   # Прозрачность для точек тепловой карты
    boundary_opacity=0.7,  # Прозрачность для поверхности границы
    level=0.5,             # Уровень решающей границы (для режимов с границей)
    limit_radius=0.5,      # Максимальный радиус тока для определения границ Re/Im и фильтрации heatmap
    v_mag_min=0,           # Минимальное значение V_mag
    v_mag_max=0.6,         # Максимальное значение V_mag
    heatmap_uncertainty_range=None
):
    """
    Строит ИНТЕРАКТИВНУЮ 3D карту решающей границы и/или тепловую карту предсказаний.
    Сетка и marching_cubes (если используется) выполняются в ДЕКАРТОВЫХ координатах тока (Re(I), Im(I), V_mag).

    Args:
        model_path (str): Путь к файлу модели (.pth).
        device (torch.device): Устройство (cpu или cuda).
        output_html_file (str): Путь для сохранения HTML.
        visualization_mode (str): Режим отображения:
            'boundary_only': Показывает только поверхность решающей границы.
            'boundary_and_fill': Показывает границу и заполняет область предсказаний > level точками.
            'heatmap': Не показывает границу, отображает точки сетки, окрашенные по значению предсказания.
        grid_density (int): Плотность сетки по каждой оси (Re, Im, V_mag).
        marker_size (float): Размер маркеров для точек заполнения или тепловой карты.
        heatmap_opacity (float): Прозрачность маркеров для режима 'heatmap'.
        boundary_opacity (float): Прозрачность поверхности границы (Mesh3d).
        level (float): Изоуровень для построения границы (используется в режимах 'boundary_only', 'boundary_and_fill').
        limit_radius (float): Максимальный модуль тока |I|, используемый для
                              определения пределов Re и Im осей и для фильтрации точек в режиме 'heatmap'.
        v_mag_min (float): Минимальное значение оси Z (V_mag).
        v_mag_max (float): Максимальное значение оси Z (V_mag).
        heatmap_uncertainty_range (tuple[float, float] | None): Диапазон предсказаний (min, max),
            точки внутри которого НЕ будут отображаться в режиме 'heatmap'.
            Если None, отображаются все точки (в пределах limit_radius).
            Пример: (0.3, 0.7) - скрыть точки с предсказаниями между 0.3 и 0.7.
    """
    print(f"--- Запуск построения 3D визуализации (Режим: {visualization_mode}) ---")
    valid_modes = ['boundary_only', 'boundary_and_fill', 'heatmap']
    if visualization_mode not in valid_modes:
        print(f"Ошибка: Неверный режим визуализации '{visualization_mode}'. Допустимые значения: {valid_modes}")
        return

    # --- 1. Загрузка модели ---
    try:
        # Убрали weights_only=False тк может вызывать проблемы совместимости,
        # если модель сохранена как state_dict, а не целиком.
        # Лучше загружать state_dict в экземпляр модели.
        # НО: если модель сохранена через torch.save(model, PATH), то False нужно.
        # Оставляем как было у автора, но с комментарием.
        # Для большей надежности, лучше передавать экземпляр класса модели, а не путь
        model = torch.load(model_path, map_location=device, weights_only=False) # weights_only=False может понадобиться
        model.eval()
        print(f"Модель {model_path} успешно загружена на {device}.")
    except FileNotFoundError:
        print(f"Ошибка: Файл модели не найден по пути: {model_path}")
        return
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # --- 2. Определение пространства и создание ДЕКАРТОВОЙ сетки ---
    re_im_limit = limit_radius # Используем радиус для пределов Re/Im
    print(f"Параметры сетки: Плотность={grid_density}, Пределы Re/Im=[-{re_im_limit}, {re_im_limit}], V_mag=[{v_mag_min}, {v_mag_max}]")

    re_range = np.linspace(-re_im_limit, re_im_limit, grid_density)
    im_range = np.linspace(-re_im_limit, re_im_limit, grid_density)
    v_mag_range = np.linspace(v_mag_min, v_mag_max, grid_density)

    # Создаем сетку в декартовых координатах
    RE_grid, IM_grid, V_mag_grid = np.meshgrid(re_range, im_range, v_mag_range, indexing='ij')
    print(f"Декартова сетка создана (форма: {RE_grid.shape})")

    # --- 3. Преобразование координат сетки для модели ---
    print("Преобразование координат декартовой сетки в полярные для модели...")
    I_mag_grid_for_model, I_angle_grid_for_model = cartesian_to_polar_current(RE_grid, IM_grid)
    print("Преобразование завершено.")

    # --- 4. Нормализация входных данных для модели ---
    print("Нормализация признаков для модели...")
    norm_I_mag, norm_I_angle, norm_V_mag = normalize_features(
        I_mag_grid_for_model, I_angle_grid_for_model, V_mag_grid
    )
    print("Нормализация завершена.")

    # --- 5. Получение предсказаний модели ---
    print("Получение предсказаний модели на сетке...")
    grid_points_flat = np.stack([
        norm_I_mag.ravel(), norm_I_angle.ravel(), norm_V_mag.ravel()
    ], axis=-1)

    grid_tensor = torch.tensor(grid_points_flat, dtype=torch.float32).to(device)

    predictions_flat = np.zeros(grid_tensor.shape[0])
    batch_size = 1024 * 32 # Увеличим батч, т.к. расчеты на CPU обычно не так ограничены памятью GPU
    with torch.no_grad():
        for i in range(0, grid_tensor.shape[0], batch_size):
            batch = grid_tensor[i:i+batch_size]
            output = model(batch)
            # Обработка вывода - предполагаем сигмоиду на выходе (0..1)
            # Если модель выдает логиты, нужно добавить torch.sigmoid()
            # output = torch.sigmoid(output) # Раскомментируйте, если модель выдает логиты
            predictions_flat[i:i+batch_size] = output.squeeze().cpu().numpy()

    prediction_grid_cartesian = predictions_flat.reshape(RE_grid.shape)
    print(f"Предсказания получены (форма: {prediction_grid_cartesian.shape})")

    # --- Инициализация переменных для графики ---
    verts, faces = None, None
    re_coords_mesh, im_coords_mesh, z_coords_mesh = None, None, None
    fill_re, fill_im, fill_z = np.array([]), np.array([]), np.array([])
    heatmap_re, heatmap_im, heatmap_z, heatmap_preds = np.array([]), np.array([]), np.array([]), np.array([])

    # --- 6. Расчет границы (если режим требует) ---
    if visualization_mode in ['boundary_only', 'boundary_and_fill']:
        print(f"\n--- Расчет границы (Marching Cubes, уровень={level}) ---")
        try:
            spacing_re = (re_range[-1] - re_range[0]) / (grid_density - 1) if grid_density > 1 else 1
            spacing_im = (im_range[-1] - im_range[0]) / (grid_density - 1) if grid_density > 1 else 1
            spacing_v = (v_mag_range[-1] - v_mag_range[0]) / (grid_density - 1) if grid_density > 1 else 1

            verts, faces, _, _ = marching_cubes(
                prediction_grid_cartesian,
                level=level,
                spacing=(spacing_re, spacing_im, spacing_v)
            )
            verts[:, 0] += re_range[0]
            verts[:, 1] += im_range[0]
            verts[:, 2] += v_mag_range[0]

            re_coords_mesh = verts[:, 0]
            im_coords_mesh = verts[:, 1]
            z_coords_mesh = verts[:, 2]
            print(f"Marching Cubes завершен. Вершин: {verts.shape[0]}, граней: {faces.shape[0]}")
            if verts.shape[0] == 0:
                 print(f"Предупреждение: Marching Cubes не нашел вершин для уровня {level}.")
                 verts, faces = None, None

        except ValueError as ve:
             print(f"Ошибка ValueError в marching_cubes: {ve}. Возможно, все значения выше или ниже уровня {level}.")
             verts, faces = None, None
        except Exception as e:
            print(f"Неожиданная ошибка при расчете границы marching_cubes: {e}")
            verts, faces = None, None

    # --- 7. Расчет точек заполнения (если режим требует) ---
    if visualization_mode == 'boundary_and_fill':
        print(f"\n--- Расчет точек заполнения (предсказания > {level}) ---")
        try:
            # Фильтруем точки сетки, где предсказание выше уровня И внутри радиуса
            fill_mask = (prediction_grid_cartesian > level) & (I_mag_grid_for_model <= limit_radius)
            fill_re = RE_grid[fill_mask]
            fill_im = IM_grid[fill_mask]
            fill_z = V_mag_grid[fill_mask]
            print(f"Найдено {fill_re.shape[0]} точек для закрашивания.")
        except Exception as e:
             print(f"Ошибка при расчете точек заполнения: {e}")

    # --- 7.5 Расчет точек для тепловой карты (если режим требует) ---
    if visualization_mode == 'heatmap':
        print(f"\n--- Расчет точек для тепловой карты (внутри радиуса {limit_radius}) ---")
        try:
            # Начальная маска: точки внутри заданного радиуса
            radius_mask = (I_mag_grid_for_model <= limit_radius)
            print(f"Начальное количество точек в радиусе: {np.sum(radius_mask)}")

            # Применяем фильтр диапазона неопределенности, если он задан
            if heatmap_uncertainty_range is not None and isinstance(heatmap_uncertainty_range, (list, tuple)) and len(heatmap_uncertainty_range) == 2:
                low_thresh, high_thresh = sorted(heatmap_uncertainty_range) # Убедимся, что порядок верный
                print(f"Применение фильтра неопределенности: исключаем точки с предсказаниями в диапазоне ({low_thresh:.3f}, {high_thresh:.3f})")
                # Маска для предсказаний ВНЕ диапазона неопределенности
                uncertainty_filter_mask = (prediction_grid_cartesian <= low_thresh) | (prediction_grid_cartesian >= high_thresh)
                # Объединяем маску радиуса и маску фильтра
                final_heatmap_mask = radius_mask & uncertainty_filter_mask
            else:
                # Если диапазон не задан, используем только маску радиуса
                final_heatmap_mask = radius_mask
                if heatmap_uncertainty_range is not None:
                    print("Предупреждение: heatmap_uncertainty_range задан некорректно, фильтрация неопределенности отключена.")

            # Применяем итоговую маску
            heatmap_re = RE_grid[final_heatmap_mask]
            heatmap_im = IM_grid[final_heatmap_mask]
            heatmap_z = V_mag_grid[final_heatmap_mask]
            heatmap_preds = prediction_grid_cartesian[final_heatmap_mask] # Берем соответствующие предсказания

            print(f"Итоговое количество точек для тепловой карты: {heatmap_re.shape[0]}")

        except Exception as e:
             print(f"Ошибка при расчете точек для тепловой карты: {e}")
             # Сбрасываем массивы в случае ошибки
             heatmap_re, heatmap_im, heatmap_z, heatmap_preds = np.array([]), np.array([]), np.array([]), np.array([])

    # --- 8. Визуализация с Plotly ---
    print("\n--- Создание 3D графика Plotly ---")
    fig = go.Figure()
    title = f"3D Визуализация Модели (Сетка {grid_density}^3)"
    show_colorbar = False # Флаг для отображения шкалы цвета

    # --- 8.1 Добавление границы Mesh3d (если есть и требуется) ---
    if visualization_mode in ['boundary_only', 'boundary_and_fill'] and verts is not None and faces is not None:
        fig.add_trace(go.Mesh3d(
            x=re_coords_mesh,
            y=im_coords_mesh,
            z=z_coords_mesh,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=boundary_opacity,
            # intensity=z_coords_mesh, # Можно окрасить по Z
            # colorscale='Viridis',
            color='lightblue', # Или фиксированный цвет
            # showscale=False, # Не нужна шкала для границы
            name=f'Решающая граница (уровень={level})'
        ))
        title = f"3D Решающая граница (Уровень {level}, Сетка {grid_density}^3)"
        print("Граница Mesh3d добавлена на график.")
    elif visualization_mode in ['boundary_only', 'boundary_and_fill']:
         print("Граница Mesh3d не будет добавлена (нет данных или режим 'heatmap').")

    # --- 8.2 Добавление точек заполнения Scatter3d (если есть и требуется) ---
    if visualization_mode == 'boundary_and_fill' and fill_re.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=fill_re, y=fill_im, z=fill_z,
            mode='markers',
            marker=dict(
                size=marker_size,
                color='orange', # Фиксированный цвет для заполнения
                opacity=0.25 # Можно сделать более прозрачным
            ),
            name=f'Область предсказания > {level}'
        ))
        title += " с заполнением"
        print("Точки заполнения Scatter3d добавлены на график.")
    elif visualization_mode == 'boundary_and_fill':
         print("Точки заполнения Scatter3d не будут добавлены (нет данных).")

    # --- 8.3 Добавление тепловой карты Scatter3d (если есть и требуется) ---
    if visualization_mode == 'heatmap' and heatmap_re.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=heatmap_re, y=heatmap_im, z=heatmap_z,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=heatmap_preds,
                colorscale='RdBu', # Красный=0, Синий=1
                opacity=heatmap_opacity,
                cmin=0, # Всегда шкала 0-1
                cmax=1,
                colorbar=dict(
                    title='Предсказание<br>(0:Красн, 1:Син)',
                    thickness=15,
                    titleside='right',
                    len=0.75,
                    yanchor='middle',
                    y=0.5,
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=['0', '0.25', '0.5', '0.75', '1']
                )
            ),
            name='Тепловая карта предсказаний'
        ))

        title_suffix = f"(Сетка {grid_density}^3, Радиус {limit_radius})"
        if heatmap_uncertainty_range is not None:
             title_suffix += f"<br>Исключен диапазон ({heatmap_uncertainty_range[0]:.2f}-{heatmap_uncertainty_range[1]:.2f})"

        title = f"3D Тепловая карта предсказаний {title_suffix}"
        show_colorbar = True
        print("Тепловая карта Scatter3d добавлена на график.")
    elif visualization_mode == 'heatmap':
        print("Тепловая карта Scatter3d не будет добавлена (нет данных или все точки отфильтрованы).")


    # --- 8.4 Настройка макета ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Re(I), ориг. ед.',
            yaxis_title='Im(I), ориг. ед.',
            zaxis_title='Модуль напряжения (V_mag), ориг. ед.',
            xaxis_range=[-re_im_limit*1.05, re_im_limit*1.05],
            yaxis_range=[-re_im_limit*1.05, re_im_limit*1.05],
            zaxis_range=[v_mag_min, v_mag_max*(1.05 if v_mag_max > 0 else 0.1)], # Чуть увеличим верхний предел Z
            aspectmode='cube' # Делает оси визуально равными
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        # showlegend=True # Легенда полезна для всех режимов
    )
    # Убираем showscale у Mesh3d, если он есть и мы показываем colorbar для heatmap
    # (Это больше не нужно, т.к. у Mesh3d теперь фиксированный цвет)
    # if show_colorbar and visualization_mode in ['boundary_only', 'boundary_and_fill']:
    #     if len(fig.data) > 0 and isinstance(fig.data[0], go.Mesh3d):
    #         fig.data[0].showscale = False

    print("Настройка макета графика завершена.")

    # --- 9. Сохранение в HTML ---
    try:
        pio.write_html(fig, output_html_file, auto_open=False) # Используем pio.write_html
        print(f"Интерактивный график сохранен в: {output_html_file}")
        print("Откройте этот файл в веб-браузере для взаимодействия.")
    except Exception as e:
        print(f"Ошибка при сохранении HTML файла: {e}")

    print(f"--- Построение 3D визуализации (Режим: {visualization_mode}) завершено ---")


# --- Запуск ---
if __name__ == "__main__":
    # Укажите путь к вашему файлу модели
    MODEL_FILE = "ML_model/trained_models/model_PDR_MLP_v2_ep33_vbl0.0052_train11.2616.pt"
    OUTPUT_HTML_BASE = 'marking_up_oscillograms/decision_boundary_v3' # Базовое имя файла

    # --- Параметры ---
    GRID_DENSITY_DEMO = 50      # Уменьшим для скорости демонстрации heatmap
    MARKER_SIZE_DEMO = 2.5
    I_RADIUS_LIMIT_DEMO = 1
    V_MIN_DEMO = 0.0
    V_MAX_DEMO = 0.6
    THRESHOLD_DEMO = 0.5 # Уровень для режимов с границей

    # Определяем устройство
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {dev}")
    # Если на CPU слишком долго, можно принудительно выбрать CPU
    # dev = torch.device("cpu")
    # print(f"Принудительно используется CPU")

    # --- Вызов для режима "граница + заполнение" ---
    print("\n*** ГЕНЕРАЦИЯ: ГРАНИЦА + ЗАПОЛНЕНИЕ ***")
    plot_3d_boundary_unified(
        model_path=MODEL_FILE,
        device=dev,
        output_html_file=f"{OUTPUT_HTML_BASE}_boundary_fill.html",
        visualization_mode='boundary_and_fill', # <--- Режим
        grid_density=GRID_DENSITY_DEMO,
        marker_size=MARKER_SIZE_DEMO,
        level=THRESHOLD_DEMO, # Используется здесь
        limit_radius=I_RADIUS_LIMIT_DEMO,
        v_mag_min=V_MIN_DEMO,
        v_mag_max=V_MAX_DEMO
    )

    # --- Вызов для режима "только граница" ---
    print("\n*** ГЕНЕРАЦИЯ: ТОЛЬКО ГРАНИЦА ***")
    plot_3d_boundary_unified(
        model_path=MODEL_FILE,
        device=dev,
        output_html_file=f"{OUTPUT_HTML_BASE}_boundary_only.html",
        visualization_mode='boundary_only', # <--- Режим
        grid_density=GRID_DENSITY_DEMO,
        marker_size=MARKER_SIZE_DEMO, # Не используется, но параметр есть
        level=THRESHOLD_DEMO, # Используется здесь
        limit_radius=I_RADIUS_LIMIT_DEMO,
        v_mag_min=V_MIN_DEMO,
        v_mag_max=V_MAX_DEMO
    )

    # --- Вызов для режима "тепловая карта" ---
    print("\n*** ГЕНЕРАЦИЯ: ТЕПЛОВАЯ КАРТА ***")
    plot_3d_boundary_unified(
        model_path=MODEL_FILE,
        device=dev,
        output_html_file=f"{OUTPUT_HTML_BASE}_heatmap.html",
        visualization_mode='heatmap', # <--- Режим
        heatmap_uncertainty_range=(0.4, 0.6), # Не показывать точки с предсказаниями между 0.3 и 0.7 (None - все точки)
        grid_density=GRID_DENSITY_DEMO, # Плотность влияет на детальность heatmap
        marker_size=MARKER_SIZE_DEMO + 0.5, # Можно сделать точки чуть крупнее для heatmap
        heatmap_opacity=0.7, # Настроим прозрачность для heatmap
        level=THRESHOLD_DEMO, # НЕ используется здесь, но параметр есть
        limit_radius=I_RADIUS_LIMIT_DEMO, # Используется для фильтрации точек
        v_mag_min=V_MIN_DEMO,
        v_mag_max=V_MAX_DEMO
    )