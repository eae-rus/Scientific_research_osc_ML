import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path
import datetime
import threading
import os
import unicodedata
import re

# --- Константы и Значения по умолчанию ---
DEFAULT_TT_PRIMARY = 1000.0
DEFAULT_TT_SECONDARY = 5.0
DEFAULT_TN_PRIMARY = 110000.0
DEFAULT_TN_SECONDARY = 100.0
DEFAULT_FREQ = 50.0

APP_TITLE = "CSV to COMTRADE Converter"

# --- Функции ---

def get_channel_type_phase(channel_name):
    """Определяет тип ('A' или 'V') и фазу ('A','B','C','N') по имени канала."""
    name_upper = channel_name.upper()
    unit = '?'
    phase = ''

    if 'I' in name_upper:
        unit = 'A'
    elif 'U' in name_upper:
        unit = 'V'
    elif 'V' in name_upper: # Дополнительная проверка на V
         unit = 'V'

    if '_A' in name_upper or name_upper.startswith('IA') or name_upper.startswith('UA') or name_upper.startswith('VA'):
        phase = 'A'
    elif '_B' in name_upper or name_upper.startswith('IB') or name_upper.startswith('UB') or name_upper.startswith('VB'):
        phase = 'B'
    elif '_C' in name_upper or name_upper.startswith('IC') or name_upper.startswith('UC') or name_upper.startswith('VC'):
        phase = 'C'
    elif '_N' in name_upper or name_upper.startswith('UN') or name_upper.startswith('IN'):
        phase = 'N'

    # Если фаза не определена явно, попытаемся извлечь из IA, UB и т.д.
    if not phase and len(channel_name) >= 2:
        potential_phase = channel_name[1]
        if potential_phase in ['A', 'B', 'C', 'N']:
            phase = potential_phase
        elif channel_name[0] in ['A', 'B', 'C', 'N'] and unit != '?': # Проверим первую букву, если вторая не фаза
             phase = channel_name[0]


    # Уточнение для 'IA', 'IB', 'IC' и 'UA', 'UB', 'UC' без суффиксов
    if phase == '' and len(channel_name) == 2 :
        if channel_name.startswith(('I','U','V')) and channel_name[1] in ('A','B','C','N'):
            phase = channel_name[1]
            if channel_name.startswith('I'): unit = 'A'
            if channel_name.startswith(('U','V')): unit = 'V'

    return unit, phase if phase else '' # Возвращаем пустую строку, если фаза не найдена

def sanitize_filename(filename, replacement='X'):
    """
    Заменяет все не-ASCII символы в строке на заданный символ.
    Также заменяет некоторые другие недопустимые для имен файлов символы Windows.
    """
    # Замена или удаление недопустимых символов для имен файлов Windows
    # Список может быть неполным, но покрывает основные случаи
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, filename)

    # Замена не-ASCII символов
    ascii_parts = []
    for char in sanitized:
        if char.isascii():
            ascii_parts.append(char)
        else:
            ascii_parts.append(replacement)
    return "".join(ascii_parts)

def process_csv_to_comtrade(csv_path, out_folder, params, status_callback):
    """Обрабатывает один CSV файл и создает CFG и DAT."""
    try:
        status_callback(f"Processing: {csv_path.name}...")
        original_base_name = csv_path.stem # Сохраняем оригинальное имя для сообщений
        # Очищаем имя файла от не-ASCII и недопустимых символов
        sanitized_base_name = sanitize_filename(original_base_name) # Используем нашу новую функцию
        # Уведомляем пользователя, если имя было изменено
        if original_base_name != sanitized_base_name:
            status_callback(f"Info: Output filename for '{original_base_name}.csv' was sanitized to '{sanitized_base_name}.cfg/.dat' due to non-ASCII or invalid characters.")
        
        cfg_path = out_folder / f"{sanitized_base_name}.cfg"
        dat_path = out_folder / f"{sanitized_base_name}.dat"

        # --- Чтение CSV ---
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV '{csv_path.name}': {e}")

        if df.empty or len(df) < 2:
            raise ValueError(f"CSV file '{csv_path.name}' is empty or has less than 2 data rows.")

        # Проверка наличия колонки времени
        if 'TimeUTC' not in df.columns:
             # Попытка найти похожую колонку
            time_col_found = None
            for col in df.columns:
                if 'TIME' in col.upper() or 'SEC' in col.upper():
                    time_col_found = col
                    break
            if not time_col_found:
                 raise ValueError(f"'TimeUTC' column not found in '{csv_path.name}' and no alternative time column identified.")
            else:
                 # Переименовываем найденную колонку для дальнейшей обработки
                 df.rename(columns={time_col_found: 'TimeUTC'}, inplace=True)
                 status_callback(f"Info: Used column '{time_col_found}' as time source for {csv_path.name}")


        # --- Определение параметров из данных ---
        channel_names = [col for col in df.columns if col != 'TimeUTC']
        num_analog_channels = len(channel_names)
        num_digital_channels = 0
        total_channels = num_analog_channels + num_digital_channels
        
        # Находим максимальные абсолютные значения для каждого канала
        max_abs_values = {}
        for ch_name in channel_names:
             # Проверяем тип данных колонки, преобразуем в числовой, если нужно
             # errors='coerce' заменит нечисловые значения на NaN
             numeric_channel_data = pd.to_numeric(df[ch_name], errors='coerce')
             # Удаляем NaN перед поиском максимума
             numeric_channel_data = numeric_channel_data.dropna()
             if numeric_channel_data.empty:
                 max_abs_values[ch_name] = 0.0 # Если все значения нечисловые или пустые
             else:
                max_abs_values[ch_name] = numeric_channel_data.abs().max()

        # Частота дискретизации
        time_diff = df['TimeUTC'].iloc[1] - df['TimeUTC'].iloc[0]
        if time_diff <= 0:
             # Попробуем следующие точки, если первая разница некорректна
             if len(df) > 2:
                 time_diff = df['TimeUTC'].iloc[2] - df['TimeUTC'].iloc[1]
             if time_diff <= 0:
                  raise ValueError(f"Cannot determine sampling rate from 'TimeUTC' in '{csv_path.name}'. Time difference is zero or negative.")
        sampling_rate = round(1.0 / time_diff)
        total_samples = len(df)

        # Временные метки
        # Используем текущее время для времени создания файла COMTRADE
        # Время начала и триггера берем из данных, если возможно, но стандарт требует дату
        now = datetime.datetime.now()
        start_time_obj = now # Используем текущее время как базовое для даты
        # Момент триггера = момент начала для простоты
        trigger_time_obj = start_time_obj

        # Форматирование времени для CFG: dd/mm/yyyy,hh:mm:ss.ffffff
        time_format = "%d/%m/%Y,%H:%M:%S.%f"
        start_time_str = start_time_obj.strftime(time_format)
        trigger_time_str = trigger_time_obj.strftime(time_format)

        # --- Генерация CFG ---
        cfg_lines = []
        # Line 1: Station name, device ID, standard revision year
        # Используем имя файла без расширения как ID устройства
        station_name = params.get('station_name', "CSV_Import") # Можно сделать настраиваемым
        device_id = sanitized_base_name
        cfg_lines.append(f"{station_name},{device_id},1999")
        # Line 2: Total channels, Analog channels (A), Digital channels (D)
        cfg_lines.append(f"{total_channels},{num_analog_channels}A,{num_digital_channels}D")

        # Analog Channel Definitions
        for i, ch_name in enumerate(channel_names):
            ch_index = i + 1
            unit, phase = get_channel_type_phase(ch_name)
            circuit_component = "" # Можно добавить логику определения (напр. 'BUS', 'CL' из имени)
            if '_BUS' in ch_name.upper(): circuit_component = 'BUS'
            elif '_CL' in ch_name.upper(): circuit_component = 'LINE' # Пример
            
            # Рассчитываем коэффициент 'a' на основе макс. значения канала
            max_val = max_abs_values[ch_name]
            # Избегаем деления на ноль, если макс. значение = 0
            # Если макс. значение 0, коэфф. 'a' не важен, но ставим 1 для корректности
            a = max_val / 32767.0 if max_val > 1e-9 else 1.0
            b = 0.0 # Смещение всегда 0 для ASCII с физическими величинами

            if unit == 'A':
                primary_factor = params['tt_primary']
                secondary_factor = params['tt_secondary']
                ps = 'P' # Предполагаем первичные значения или шкалу
            elif unit == 'V':
                primary_factor = params['tn_primary']
                secondary_factor = params['tn_secondary']
                ps = 'P'
            else: # Неизвестный тип
                # Используем настройки тока по умолчанию для факторов, но 'a' уже посчитан
                primary_factor = params['tt_primary']
                secondary_factor = params['tt_secondary']
                ps = 'P'
                unit = '?'
                
            # Устанавливаем флаг 'ps' в зависимости от выбора пользователя
            if params['data_type'] == "Первичные":
                ps = 'P'
            elif params['data_type'] == "Вторичные":
                ps = 'S'
            else: # На всякий случай, если значение некорректно
                ps = 'P' # По умолчанию первичные

            skew = 0.0
            min_digital = -32768
            max_digital = 32767

            # Формат строки: n,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,ps
            cfg_lines.append(f"{ch_index},{ch_name},{phase},{circuit_component},{unit},{a},{b},{skew},{min_digital},{max_digital},{primary_factor},{secondary_factor},{ps}")

        # Digital Channel Definitions (none in this case)

        # Network frequency
        cfg_lines.append(f"{params['net_freq']:.1f}")
        # Number of sampling rates (only 1)
        cfg_lines.append("1")
        # Sampling rate info: samp, endsamp
        cfg_lines.append(f"{sampling_rate},{total_samples}")
        # Start date/time
        cfg_lines.append(start_time_str)
        # Trigger date/time
        cfg_lines.append(trigger_time_str)
        # Data file type
        cfg_lines.append("ASCII")
        # Timestamp multiplication factor (optional, default is 1)
        cfg_lines.append("1") # Добавлено для полноты

        # Запись CFG файла
        try:
            cfg_content_str = '\n'.join(cfg_lines) + '\n'
            cfg_content_bytes = cfg_content_str.encode('ascii') # Содержимое все еще должно быть ASCII
            with open(cfg_path, 'wb') as f_cfg:
                f_cfg.write(cfg_content_bytes)
        except UnicodeEncodeError:
            error_message = f"Error: Non-ASCII characters found in generated CFG CONTENT for {csv_path.name}. Cannot save as COMTRADE ASCII."
            status_callback(error_message)
            print(f"ERROR: {error_message}")
            return False

        # --- Генерация DAT ---
        dat_lines = []
        for i, row in df.iterrows():
            sample_num = i + 1
            # Время в микросекундах от начала записи
            timestamp_us = int(row['TimeUTC'] * 1_000_000)
            # Значения каналов - форматируем в строку с запятыми
            # Используем .get(ch, '') чтобы обработать случайные пропуски колонок, если вдруг
            channel_values = [f"{row.get(ch, 0):.6f}" for ch in channel_names] # Формат с 6 знаками после запятой
            dat_lines.append(f"{sample_num},{timestamp_us},{','.join(channel_values)}")

        # Запись DAT файла (используем бинарный режим 'wb' для надежности)
        try:
            dat_content_str = '\n'.join(dat_lines) + '\n'
            dat_content_bytes = dat_content_str.encode('ascii') # Содержимое все еще должно быть ASCII
            with open(dat_path, 'wb') as f_dat:
                f_dat.write(dat_content_bytes)
        except UnicodeEncodeError:
            error_message = f"Error: Non-ASCII characters found in generated DAT CONTENT for {csv_path.name}. Cannot save as COMTRADE ASCII."
            status_callback(error_message)
            print(f"ERROR: {error_message}")
            return False

        # Сообщение об успехе использует оригинальное имя CSV и очищенное имя COMTRADE
        status_callback(f"Successfully converted: {csv_path.name} -> {sanitized_base_name}.cfg/.dat")
        return True

    except Exception as e:
        # Используем оригинальное имя CSV в сообщении об ошибке
        error_message = f"Error processing {csv_path.name}: {e}"
        status_callback(error_message)
        import traceback
        print(f"ERROR processing {csv_path.name}:")
        traceback.print_exc()
        return False


def start_conversion_thread(app_instance):
    """Запускает конвертацию в отдельном потоке, чтобы не блокировать GUI."""
    # Блокируем кнопку запуска
    app_instance.run_button.config(state=tk.DISABLED)
    app_instance.status_label.config(text="Starting conversion...")
    app_instance.progress_bar['value'] = 0

    # Считываем параметры из GUI
    try:
        params = {
            'tt_primary': float(app_instance.tt_primary_entry.get()),
            'tt_secondary': float(app_instance.tt_secondary_entry.get()),
            'tn_primary': float(app_instance.tn_primary_entry.get()),
            'tn_secondary': float(app_instance.tn_secondary_entry.get()),
            'net_freq': float(app_instance.freq_entry.get()),
            'data_type': app_instance.data_type_var.get(),
        }
        input_folder = app_instance.input_folder_var.get()
        output_folder = app_instance.output_folder_var.get()

        if not input_folder or not Path(input_folder).is_dir():
            messagebox.showerror("Error", "Please select a valid input folder.")
            app_instance.run_button.config(state=tk.NORMAL)
            app_instance.status_label.config(text="Ready.")
            return
        if not output_folder or not Path(output_folder).is_dir():
             # Попытаемся создать папку, если она не существует
             try:
                 Path(output_folder).mkdir(parents=True, exist_ok=True)
                 if not Path(output_folder).is_dir(): # Проверка после попытки создания
                      raise OSError("Could not create output directory.")
             except Exception as e:
                messagebox.showerror("Error", f"Please select or create a valid output folder.\nError: {e}")
                app_instance.run_button.config(state=tk.NORMAL)
                app_instance.status_label.config(text="Ready.")
                return

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values for all parameters.")
        app_instance.run_button.config(state=tk.NORMAL)
        app_instance.status_label.config(text="Ready.")
        return

    # Запуск обработки в отдельном потоке
    thread = threading.Thread(target=run_conversion, args=(app_instance, input_folder, output_folder, params), daemon=True)
    thread.start()

def run_conversion(app_instance, input_folder, output_folder, params):
    """Основная функция конвертации, выполняемая в потоке."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    csv_files = list(input_path.glob('*.csv'))

    if not csv_files:
        app_instance.update_status("No CSV files found in the input folder.")
        app_instance.run_button.config(state=tk.NORMAL)
        return

    total_files = len(csv_files)
    app_instance.progress_bar['maximum'] = total_files
    files_processed = 0
    files_succeeded = 0

    for i, csv_file in enumerate(csv_files):
        success = process_csv_to_comtrade(csv_file, output_path, params, app_instance.update_status)
        files_processed += 1
        if success:
             files_succeeded += 1
        # Обновляем прогресс бар в главном потоке через schedule
        app_instance.master.after(0, app_instance.update_progress, i + 1)

    final_message = f"Conversion finished. Processed: {files_processed}, Succeeded: {files_succeeded}, Failed: {files_processed - files_succeeded}."
    app_instance.update_status(final_message)
    # Разблокируем кнопку по завершении
    app_instance.run_button.config(state=tk.NORMAL)
    messagebox.showinfo("Complete", final_message)


# --- Класс GUI ---
class ConverterApp:
    def __init__(self, master):
        self.master = master
        master.title(APP_TITLE)
        master.geometry("550x450") # Немного увеличим размер

        # Переменные для хранения путей
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        
        # Переменная для хранения выбора типа данных
        self.data_type_var = tk.StringVar(value="Первичные") # Значение по умолчанию

        # Фрейм для параметров
        param_frame = ttk.LabelFrame(master, text="Parameters", padding=(10, 5))
        param_frame.pack(padx=10, pady=10, fill=tk.X)

        # --- Параметры ТТ ---
        ttk.Label(param_frame, text="TT Primary (A):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.tt_primary_entry = ttk.Entry(param_frame, width=15)
        self.tt_primary_entry.insert(0, str(DEFAULT_TT_PRIMARY))
        self.tt_primary_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="TT Secondary (A):").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.tt_secondary_entry = ttk.Entry(param_frame, width=15)
        self.tt_secondary_entry.insert(0, str(DEFAULT_TT_SECONDARY))
        self.tt_secondary_entry.grid(row=0, column=3, padx=5, pady=2)

        # --- Параметры ТН ---
        ttk.Label(param_frame, text="TN Primary (V):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.tn_primary_entry = ttk.Entry(param_frame, width=15)
        self.tn_primary_entry.insert(0, str(DEFAULT_TN_PRIMARY))
        self.tn_primary_entry.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="TN Secondary (V):").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.tn_secondary_entry = ttk.Entry(param_frame, width=15)
        self.tn_secondary_entry.insert(0, str(DEFAULT_TN_SECONDARY))
        self.tn_secondary_entry.grid(row=1, column=3, padx=5, pady=2)

        # --- Частота сети ---
        ttk.Label(param_frame, text="Network Freq (Hz):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.freq_entry = ttk.Entry(param_frame, width=15)
        self.freq_entry.insert(0, str(DEFAULT_FREQ))
        self.freq_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # --- Выбор типа данных в CSV ---
        ttk.Label(param_frame, text="Data in CSV:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        data_type_options = ["Первичные", "Вторичные"]
        self.data_type_combo = ttk.Combobox(param_frame, textvariable=self.data_type_var, values=data_type_options, state="readonly", width=12)
        self.data_type_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        # Можно также использовать Radiobutton, если предпочитаете:
        #ttk.Radiobutton(param_frame, text="Первичные", variable=self.data_type_var, value="Первичные").grid(row=3, column=1, sticky=tk.W)
        #ttk.Radiobutton(param_frame, text="Вторичные", variable=self.data_type_var, value="Вторичные").grid(row=3, column=2, sticky=tk.W)

        # Фрейм для выбора папок
        folder_frame = ttk.LabelFrame(master, text="Folders", padding=(10, 5))
        folder_frame.pack(padx=10, pady=5, fill=tk.X)

        # --- Выбор входной папки ---
        ttk.Label(folder_frame, text="Input CSV Folder:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.input_entry = ttk.Entry(folder_frame, textvariable=self.input_folder_var, width=40)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        self.input_button = ttk.Button(folder_frame, text="Browse...", command=self.select_input_folder)
        self.input_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Выбор выходной папки ---
        ttk.Label(folder_frame, text="Output COMTRADE Folder:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_entry = ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=40)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        self.output_button = ttk.Button(folder_frame, text="Browse...", command=self.select_output_folder)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)

        # Фрейм для запуска и прогресса
        run_frame = ttk.Frame(master, padding=(10, 10))
        run_frame.pack(padx=10, pady=10, fill=tk.X)

        # --- Кнопка Запуска ---
        self.run_button = ttk.Button(run_frame, text="Start Conversion", command=lambda: start_conversion_thread(self))
        self.run_button.pack(pady=5)

        # --- Прогресс Бар ---
        self.progress_bar = ttk.Progressbar(run_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        # --- Статус ---
        self.status_label = ttk.Label(run_frame, text="Ready. Select folders and parameters.")
        self.status_label.pack(pady=5)

        # Установка текущей директории как начальной для выходной папки (опционально)
        self.output_folder_var.set(os.getcwd())


    def select_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.input_folder_var.set(folder_selected)

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_var.set(folder_selected)

    def update_status(self, message):
        # Обновление статуса из другого потока
        self.master.after(0, self._set_status, message)

    def _set_status(self, message):
         # Этот метод вызывается в главном потоке
        self.status_label.config(text=message)

    def update_progress(self, value):
        # Этот метод вызывается в главном потоке
        self.progress_bar['value'] = value
        self.master.update_idletasks() # Обновить GUI немедленно

# --- Запуск приложения ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()