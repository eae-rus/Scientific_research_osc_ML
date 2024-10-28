В "Name_signals" - приведены все созданные универсальные названия разделённые по группам.
В "universal_analog_signals_name v2" сохранены соотнесения названия с универсальными кодами.

Названия получены на основе имени сигнала + фазы сигнала + едиинцы измерения сигнала.
Применялся следующий код:

with open(file_path, 'r', encoding='utf-8') as file:
	lines = file.readlines() # чтение cfg файла
	analog_signal = lines[2 + i].split(',') # получаем аналоговый сигнал
    name, phase, unit = analog_signal[1], analog_signal[2], analog_signal[4] # получаем название, фазу и единицу измерения
    name, phase, unit = name.replace(' ', ''), phase.replace(' ', ''), unit.replace(' ', '') # удаляем пробелы
    signal_name = name + ' | phase:' + phase + ' | unit:' + unit # создаем комбинированное название сигнала