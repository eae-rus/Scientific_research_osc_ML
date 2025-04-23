import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

try:
    from skimage.measure import marching_cubes
except ImportError:
    print("Пожалуйста, установите scikit-image: pip install scikit-image")
    exit()

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
    grid_density=60,       # Плотность сетки (теперь одна для всего)
    show_fill=True,        # Показывать ли точки заполнения
    fill_marker_size=2.5,  # Размер маркеров для заполнения
    level=0.5,             # Уровень решающей границы
    limit_radius=0.5,      # Максимальный радиус тока для определения границ Re/Im
    v_mag_min=0,           # Минимальное значение V_mag
    v_mag_max=0.6          # Максимальное значение V_mag
):
    """
    Строит ИНТЕРАКТИВНУЮ 3D карту решающей границы и (опционально) заполнения.
    Сетка и marching_cubes выполняются в ДЕКАРТОВЫХ координатах тока (Re(I), Im(I), V_mag).

    Args:
        model_path (str): Путь к файлу модели (.pth).
        device (torch.device): Устройство (cpu или cuda).
        output_html_file (str): Путь для сохранения HTML.
        grid_density (int): Плотность сетки по каждой оси (Re, Im, V_mag).
        show_fill (bool): Если True, показывает область предсказаний > level точками.
        fill_marker_size (float): Размер маркеров для точек заполнения.
        level (float): Изоуровень для построения границы (обычно 0.5).
        limit_radius (float): Максимальный модуль тока |I|, используемый для
                              определения пределов Re и Im осей (-limit_radius до +limit_radius).
        v_mag_min (float): Минимальное значение оси Z (V_mag).
        v_mag_max (float): Максимальное значение оси Z (V_mag).
    """
    print("--- Запуск построения 3D границы (Единая функция) ---")
    # --- 1. Загрузка модели ---
    try:
        model = torch.load(model_path, map_location=device)
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
    # Для каждой точки (Re, Im) декартовой сетки вычисляем полярные (I_mag, I_angle)
    print("Преобразование координат декартовой сетки в полярные для модели...")
    I_mag_grid_for_model, I_angle_grid_for_model = cartesian_to_polar_current(RE_grid, IM_grid)
    # V_mag_grid_for_model это просто V_mag_grid
    print("Преобразование завершено.")

    # --- 4. Нормализация входных данных для модели ---
    print("Нормализация признаков для модели...")
    norm_I_mag, norm_I_angle, norm_V_mag = normalize_features(
        I_mag_grid_for_model, I_angle_grid_for_model, V_mag_grid # Используем V_mag_grid напрямую
    )
    print("Нормализация завершена.")

    # --- 5. Получение предсказаний модели ---
    print("Получение предсказаний модели на сетке...")
    # Собираем нормализованные данные в один массив для подачи в модель
    grid_points_flat = np.stack([
        norm_I_mag.ravel(), norm_I_angle.ravel(), norm_V_mag.ravel()
    ], axis=-1)
    grid_tensor = torch.tensor(grid_points_flat, dtype=torch.float32).to(device)

    predictions_flat = np.zeros(grid_tensor.shape[0])
    batch_size = 1024 * 16 # Можно настроить размер батча
    with torch.no_grad():
        for i in range(0, grid_tensor.shape[0], batch_size):
            batch = grid_tensor[i:i+batch_size]
            output = model(batch)
            predictions_flat[i:i+batch_size] = output.squeeze().cpu().numpy()

    # Возвращаем предсказания к форме исходной ДЕКАРТОВОЙ сетки
    prediction_grid_cartesian = predictions_flat.reshape(RE_grid.shape)
    print(f"Предсказания получены (форма: {prediction_grid_cartesian.shape})")

    # --- 6. Расчет границы с помощью Marching Cubes ---
    verts, faces = None, None
    re_coords_mesh, im_coords_mesh, z_coords_mesh = None, None, None
    print(f"\n--- Расчет границы (Marching Cubes на декартовой сетке, уровень={level}) ---")
    try:
        # Расчет шага сетки для marching_cubes
        spacing_re = (re_range[-1] - re_range[0]) / (grid_density - 1) if grid_density > 1 else 1
        spacing_im = (im_range[-1] - im_range[0]) / (grid_density - 1) if grid_density > 1 else 1
        spacing_v = (v_mag_range[-1] - v_mag_range[0]) / (grid_density - 1) if grid_density > 1 else 1

        # Применяем marching_cubes к сетке предсказаний (которая имеет структуру декартовой сетки)
        verts, faces, _, _ = marching_cubes(
            prediction_grid_cartesian,
            level=level,
            spacing=(spacing_re, spacing_im, spacing_v) # Указываем шаг по Re, Im, V_mag
        )

        # Корректируем координаты вершин, добавляя начальные значения осей
        verts[:, 0] += re_range[0] # Добавляем Re_min
        verts[:, 1] += im_range[0] # Добавляем Im_min
        verts[:, 2] += v_mag_range[0] # Добавляем V_mag_min

        # Теперь verts содержит координаты в Re, Im, V_mag
        re_coords_mesh = verts[:, 0]
        im_coords_mesh = verts[:, 1]
        z_coords_mesh = verts[:, 2]
        print(f"Marching Cubes завершен. Вершин: {verts.shape[0]}, граней: {faces.shape[0]}")
        if verts.shape[0] == 0:
             print("Предупреждение: Marching Cubes не нашел вершин для уровня {level}. Граница не будет построена.")
             verts, faces = None, None # Убедимся, что они None

    except ValueError as ve:
         print(f"Ошибка ValueError в marching_cubes: {ve}")
         print("Возможно, массив предсказаний полностью выше или ниже уровня {level}.")
         verts, faces = None, None
    except Exception as e:
        print(f"Неожиданная ошибка при расчете границы marching_cubes: {e}")
        verts, faces = None, None


    # --- 7. Расчет точек заполнения (если требуется) ---
    fill_re, fill_im, fill_z = np.array([]), np.array([]), np.array([]) # Инициализация
    if show_fill:
        print(f"\n--- Расчет точек заполнения (предсказания > {level}) ---")
        try:
            # Фильтруем точки ИСХОДНОЙ ДЕКАРТОВОЙ сетки, где предсказание выше уровня
            # Можно добавить доп. условие на радиус, если нужно отсечь углы квадрата Re/Im
            # fill_mask = (prediction_grid_cartesian > level) & (I_mag_grid_for_model <= limit_radius)
            fill_mask = (prediction_grid_cartesian > level)

            fill_re = RE_grid[fill_mask]
            fill_im = IM_grid[fill_mask]
            fill_z = V_mag_grid[fill_mask]
            print(f"Найдено {fill_re.shape[0]} точек для закрашивания.")
        except Exception as e:
             print(f"Ошибка при расчете точек заполнения: {e}")


    # --- 8. Визуализация с Plotly ---
    print("\n--- Создание 3D графика Plotly ---")
    fig = go.Figure()

    # --- 8.1 Добавление границы Mesh3d (если есть) ---
    if verts is not None and faces is not None:
        fig.add_trace(go.Mesh3d(
            x=re_coords_mesh,
            y=im_coords_mesh,
            z=z_coords_mesh,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.7, # Можно настроить прозрачность
            intensity=z_coords_mesh, # Окрашиваем по Z (V_mag)
            colorscale='Viridis',
            showscale= not show_fill, # Показываем шкалу, только если нет заполнения (чтобы не мешала)
            name=f'Решающая граница (уровень={level})'
        ))
        print("Граница Mesh3d добавлена на график.")
    else:
         print("Граница Mesh3d не будет добавлена (нет данных).")

    # --- 8.2 Добавление точек заполнения Scatter3d (если есть и требуется) ---
    if show_fill and fill_re.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=fill_re, y=fill_im, z=fill_z,
            mode='markers',
            marker=dict(
                size=fill_marker_size,
                color='orange', # Цвет заполнения
                opacity=0.15   # Прозрачность заполнения
            ),
            name=f'Область предсказания > {level}'
        ))
        print("Точки заполнения Scatter3d добавлены на график.")
    elif show_fill:
         print("Точки заполнения Scatter3d не будут добавлены (нет данных).")

    # --- 8.3 Настройка макета ---
    title = f"3D Решающая граница (Декартова сетка {grid_density}^3)"
    if show_fill:
        title += " с заполнением"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Re(I), ориг. ед.',
            yaxis_title='Im(I), ориг. ед.',
            zaxis_title='Модуль напряжения (V_mag), ориг. ед.',
            # Устанавливаем пределы осей чуть шире для наглядности
            xaxis_range=[-re_im_limit*1.05, re_im_limit*1.05],
            yaxis_range=[-re_im_limit*1.05, re_im_limit*1.05],
            zaxis_range=[v_mag_min, v_mag_max],
            # Соотношение сторон, чтобы круг Re/Im выглядел круглым
            # aspectratio=dict(x=1, y=1, z=max(0.1, (v_mag_max-v_mag_min)/(2*re_im_limit)) if re_im_limit > 0 else 1),
            # aspectmode='manual' # Используем ручные пропорции
            aspectmode='cube' # Автоматически делает оси визуально равными
        ),
        margin=dict(l=0, r=0, b=0, t=50), # Отступы
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Положение легенды
    )
    print("Настройка макета графика завершена.")

    # --- 9. Сохранение в HTML ---
    try:
        fig.write_html(output_html_file, auto_open=False)
        print(f"Интерактивный график сохранен в: {output_html_file}")
        print("Откройте этот файл в веб-браузере для взаимодействия.")
    except Exception as e:
        print(f"Ошибка при сохранении HTML файла: {e}")

    print("--- Построение 3D границы завершено ---")

# --- Запуск ---
if __name__ == "__main__":
    # Укажите путь к вашему файлу модели
    MODEL_FILE = 'model_PDR_MLP_v1_ep33_vbl0.0094_train12.8831.pt'
    # MODEL_FILE = 'model_PDR_MLP_v1_ep12_vbl0.3813_train588.8274.pt'
    OUTPUT_HTML = 'decision_boundary_mesh_cartesian_fill_scatter_3d.html'
    # Параметры для новой функции
    GRID_DENSITY = 30      # Плотность сетки (влияет на качество и время расчета)
    SHOW_FILL_POINTS = False # True - показывать оранжевые точки, False - только границу
    MARKER_SIZE = 2.0      # Размер точек заполнения
    I_RADIUS_LIMIT = 0.4   # Максимальный радиус тока для осей Re/Im
    V_MIN = 0.0
    V_MAX = 0.6

    # Определяем устройство
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Используется GPU")
    else:
        dev = torch.device("cpu")
        print("Используется CPU")

    # Вызов объединенной функции
    plot_3d_boundary_unified(
        model_path=MODEL_FILE,
        device=dev,
        output_html_file=OUTPUT_HTML,
        grid_density=GRID_DENSITY,
        show_fill=SHOW_FILL_POINTS,
        fill_marker_size=MARKER_SIZE,
        limit_radius=I_RADIUS_LIMIT,
        v_mag_min=V_MIN,
        v_mag_max=V_MAX
        # level=0.5 # Можно оставить по умолчанию
    )