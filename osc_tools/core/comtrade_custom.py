# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Evdakov Aleksey Evgenievich
# Создано на основе библиотеки "comtrade 0.1.2" (Copyright (c) 2018 David Rodrigues Parrini) предоставленной по ссылке "https://pypi.org/project/comtrade/"

# Данная лицензия разрешает лицам, получившим копию данного программного обеспечения и сопутствующей документации
# (далее – Программное обеспечение), безвозмездно использовать Программное обеспечение без ограничений, включая, но не ограничиваясь,
# правами на использование, копирование, изменение, слияние, публикацию, распространение, сублицензирование и/или продажу
# копий Программного обеспечения, а также лицам, которым предоставляется данное Программное обеспечение, при соблюдении следующих условий:

# Вышеуказанное уведомление об авторском праве и данное уведомление о разрешении должны быть включены во все
# копии или существенные части Программного обеспечения.

# ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ ПРЕДОСТАВЛЯЕТСЯ «КАК ЕСТЬ», БЕЗ КАКИХ-ЛИБО ГАРАНТИЙ, ЯВНЫХ ИЛИ ПОДРАЗУМЕВАЕМЫХ, ВКЛЮЧАЯ, НО НЕ ОГРАНИЧИВАЯСЬ,
# ГАРАНТИИ ТОВАРНОЙ ПРИГОДНОСТИ, СООТВЕТСТВИЯ ОПРЕДЕЛЕННОМУ НАЗНАЧЕНИЮ И ОТСУТСТВИЯ НАРУШЕНИЙ. НИ В КАКОМ СЛУЧАЕ
# АВТОРЫ ИЛИ ПРАВООБЛАДАТЕЛИ НЕ НЕСУТ ОТВЕТСТВЕННОСТИ ПО КАКИМ-ЛИБО ИСКАМ, УБЫТКАМ ИЛИ ДРУГИМ ТРЕБОВАНИЯМ, БУДЬ ТО
# В РЕЗУЛЬТАТЕ ДЕЙСТВИЯ ДОГОВОРА, ДЕЛИКТА ИЛИ ИНОГО, ВОЗНИКШИМ ИЗ, ВНЕ ИЛИ В СВЯЗИ С ПРОГРАММНЫМ ОБЕСПЕЧЕНИЕМ ИЛИ
# ИСПОЛЬЗОВАНИЕМ ИЛИ ИНЫМИ ДЕЙСТВИЯМИ С ПРОГРАММНЫМ ОБЕСПЕЧЕНИЕМ.



import array
import datetime as dt
import errno
import io
import math
import os
import re
import struct
import sys
import warnings
import numpy as np
import pandas as pd

try:
    import numpy
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False


# Ревизии стандарта COMTRADE
REV_1991 = "1991"
REV_1999 = "1999"
REV_2013 = "2013"

# Типы форматов файлов DAT
TYPE_ASCII = "ASCII"
TYPE_BINARY = "BINARY"
TYPE_BINARY32 = "BINARY32"
TYPE_FLOAT32 = "FLOAT32"

# Специальные значения
TIMESTAMP_MISSING = 0xFFFFFFFF

# Заголовки CFF
CFF_HEADER_REXP = r"(?i)--- file type: ([a-z]+)(?:\s+([a-z0-9]+)(?:\s*\:\s*([0-9]+))?)? ---$"

# общий символ-разделитель полей данных файлов CFG и ASCII DAT
SEPARATOR = ","

# регулярное выражение для временной метки
re_date = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{2,4})")
re_time = re.compile(r"([0-9]{1,2}):([0-9]{2}):([0-9]{2})(\.([0-9]{1,12}))?")


# Предупреждение о нестандартной ревизии
WARNING_UNKNOWN_REVISION = "Неизвестная ревизия стандарта \"{}\""
# Предупреждение о дате и времени с наносекундным разрешением
WARNING_DATETIME_NANO = "Неподдерживаемые объекты datetime с наносекундным \
разрешением. Используются усеченные значения."
# Дата и время с годом 0, месяцем 0 и/или днем 0.
WARNING_MINDATE = "Отсутствуют значения даты. Используются минимальные значения: {}."


def _read_sep_values(line, expected: int = -1, default: str = ''):
    values = tuple(map(lambda cell: cell.strip(), line.split(SEPARATOR)))
    if expected == -1 or len(values) == expected:
        return values
    return [values[i] if i < len(values) else default
            for i in range(expected)]


def _preallocate_values(array_type, size, use_numpy_arrays):
    type_mapping_numpy = {"f": "float32", "i": "int32"}
    if HAS_NUMPY and use_numpy_arrays:
        return numpy.zeros(size, dtype=type_mapping_numpy[array_type])
    return array.array(array_type, [0]) * size


def _prevent_null(str_value: str, value_type: type, default_value):
    if len(str_value.strip()) == 0:
        return default_value
    else:
        return value_type(str_value)


def _get_date(date_str: str) -> tuple:
    m = re_date.match(date_str)
    if m is not None:
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3))
        return day, month, year
    return 0, 0, 0


def _get_time(time_str: str, ignore_warnings: bool = False) -> tuple:
    m = re_time.match(time_str)
    if m is not None:
        hour = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))
        fracsec_str = m.group(5)
        # Дополнить дробную часть секунд нулями справа
        if len(fracsec_str) <= 6:
            fracsec_str = fill_with_zeros_to_the_right(fracsec_str, 6)
        else:
            fracsec_str = fill_with_zeros_to_the_right(fracsec_str, 9)

        frac_second = int(fracsec_str)
        in_nanoseconds = len(fracsec_str) > 6
        microsecond = frac_second

        if in_nanoseconds:
            # Разрешение в наносекундах не поддерживается модулем datetime,
            # поэтому ниже оно преобразуется в целое число.
            if not ignore_warnings:
                warnings.warn(Warning(WARNING_DATETIME_NANO))
            microsecond = int(microsecond * 1E-3)
        return hour, minute, second, microsecond, in_nanoseconds


def fill_with_zeros_to_the_right(number_str: str, width: int):
    actual_len = len(number_str)
    if actual_len < width:
        difference = width - actual_len
        fill_chars = "0"*difference
        return number_str + fill_chars
    return number_str


def _read_timestamp(timestamp_line: str, rev_year: str, ignore_warnings: bool = False) -> tuple:
    """Обрабатывает поля, разделенные запятыми, и возвращает кортеж, содержащий временную метку
    и логическое значение, указывающее, используются ли наносекунды.
    Может возвращать временную метку 00/00/0000 00:00:00.000 для пустых строк
    или пустых пар."""
    day, month, year, hour, minute, second, microsecond = (0,)*7
    nanosec = False
    if len(timestamp_line.strip()) > 0:
        values = _read_sep_values(timestamp_line, 2)
        if len(values) >= 2:
            date_str, time_str = values[0:2]
            if len(date_str.strip()) > 0:
                # Формат 1991 года использует формат мм/дд/гггг
                if rev_year == REV_1991:
                    month, day, year = _get_date(date_str)
                # Современные форматы используют формат дд/мм/гггг
                else:
                    day, month, year = _get_date(date_str)
            if len(time_str.strip()) > 0:
                hour, minute, second, microsecond, \
                    nanosec = _get_time(time_str, ignore_warnings)

    using_min_data = False
    if year <= 0:
        year = dt.MINYEAR
        using_min_data = True
    if month <= 0:
        month = 1
        using_min_data = True
    if day <= 0:
        day = 1
        using_min_data = True
    # Информация о часовом поясе не поддерживается
    tzinfo = None
    timestamp = dt.datetime(year, month, day, hour, minute, second,
                            microsecond, tzinfo)
    if not ignore_warnings and using_min_data:
        warnings.warn(Warning(WARNING_MINDATE.format(str(timestamp))))
    return timestamp, nanosec


def _file_is_utf8(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return _stream_is_utf8(file)
    return False


def _stream_is_utf8(stream):
    try:
        contents = stream.readlines()
    except UnicodeDecodeError as exception:
        return True
    return False


class Cfg:
    """Разбирает и хранит данные CFG из Comtrade."""
    # единицы временной базы
    TIME_BASE_NANOSEC = 1E-9
    TIME_BASE_MICROSEC = 1E-6

    def __init__(self, **kwargs):
        """
        Конструктор объекта Cfg.

        Аргументы ключевого слова:
        ignore_warnings -- выводить ли предупреждения в stdout
            (по умолчанию: False)
        """
        self.filename = ""
        # неявные данные
        self._time_base = self.TIME_BASE_MICROSEC

        # Данные CFG по умолчанию
        self._station_name = ""
        self._rec_dev_id = ""
        self._rev_year = 2013
        self._channels_count = 0
        self._analog_channels = []
        self._status_channels = []
        self._analog_count = 0
        self._status_count = 0
        self._frequency = 0.0
        self._nrates = 1
        self._sample_rates = []
        self._timestamp_critical = False
        self._start_timestamp = dt.datetime(1900, 1, 1)
        self._trigger_timestamp = dt.datetime(1900, 1, 1)
        self._ft = TYPE_ASCII
        self._time_multiplier = 1.0
        # информация о ревизии стандарта 2013 года
        # time_code,local_code = 0,0 означает, что местное время - UTC
        self._time_code = 0
        self._local_code = 0
        # tmq_code,leapsec
        self._tmq_code = 0
        self._leap_second = 0

        if "ignore_warnings" in kwargs:
            self.ignore_warnings = kwargs["ignore_warnings"]
        else:
            self.ignore_warnings = False

    @property
    def station_name(self) -> str:
        """Возвращает имя станции записывающего устройства."""
        return self._station_name
    
    @property
    def rec_dev_id(self) -> str:
        """Возвращает идентификатор записывающего устройства."""
        return self._rec_dev_id

    @property
    def rev_year(self) -> int:
        """Возвращает год ревизии COMTRADE."""
        return self._rev_year

    @property
    def channels_count(self) -> int:
        """Возвращает общее количество каналов."""
        return self._channels_count
    
    @channels_count.setter
    def channels_count(self, count_corrector) -> None:
        """Устанавливает общее количество каналов."""
        self._channels_count = count_corrector

    @property
    def analog_channels(self) -> list:
        """Возвращает список аналоговых каналов с полным описанием."""
        return self._analog_channels

    @property
    def status_channels(self) -> list:
        """Возвращает список дискретных каналов с полным описанием."""
        return self._status_channels
    
    @property
    def analog_count(self) -> int:
        """Возвращает количество аналоговых каналов."""
        return self._analog_count
    
    @analog_count.setter
    def analog_count(self, count_corrector) -> None:
        """Устанавливает количество аналоговых каналов."""
        self._analog_count = count_corrector
    
    @property
    def status_count(self) -> int:
        """Возвращает количество дискретных каналов."""
        return self._status_count
    
    @status_count.setter
    def status_count(self, count_corrector) -> None:
        """Устанавливает количество дискретных каналов."""
        self._status_count = count_corrector
    
    @property
    def time_base(self) -> float:
        """Возвращает временную базу."""
        return self._time_base

    @property
    def frequency(self) -> float:
        """Возвращает измеренную частоту сети в герцах."""
        return self._frequency
    
    @property
    def ft(self) -> str:
        """Возвращает ожидаемый формат файла DAT."""
        return self._ft
    
    @property
    def timemult(self) -> float:
        """Возвращает множитель времени DAT (по умолчанию = 1)."""
        return self._time_multiplier

    @property
    def timestamp_critical(self) -> bool:
        """Возвращает, должен ли файл DAT содержать ненулевые
         значения временных меток."""
        return self._timestamp_critical
    
    @property
    def start_timestamp(self) -> dt.datetime:
        """Возвращает временную метку начала записи в виде объекта datetime."""
        return self._start_timestamp
    
    @property
    def trigger_timestamp(self) -> dt.datetime:
        """Возвращает временную метку триггера в виде объекта datetime."""
        return self._trigger_timestamp
    
    @property
    def nrates(self) -> int:
        """Возвращает количество различных частот дискретизации в файле DAT."""
        return self._nrates
    
    @property
    def sample_rates(self) -> list:
        """
        Возвращает список пар, описывающих количество выборок для данной
        частоты дискретизации.
        """ 
        return self._sample_rates

    # Устаревшие свойства - заменено "Digital" на "Status"
    @property
    def digital_channels(self) -> list:
        """Возвращает список двумерных значений дискретных каналов."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_channels устарело, "
                                        "используйте вместо него status_channels."))
        return self._status_channels

    @property
    def digital_count(self) -> int:
        """Возвращает количество дискретных каналов."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_count устарело, "
                                        "используйте вместо него status_count."))
        return self._status_count
    
    def load(self, filepath, **user_kwargs):
        """Загружает и читает содержимое файла CFG."""
        self.filepath = filepath

        if os.path.isfile(self.filepath):
            kwargs = {}
            if "encoding" not in user_kwargs and _file_is_utf8(self.filepath):
                kwargs["encoding"] = "utf-8"
            elif "encoding" in user_kwargs:
                kwargs["encoding"] = user_kwargs["encoding"]
            with open(self.filepath, "r", **kwargs) as cfg:
                self._read_io(cfg)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 
                                    self.filepath)

    def read(self, cfg_lines):
        """Читает данные в формате CFG из объекта FileIO или StringIO."""
        if type(cfg_lines) is str:
            self._read_io(io.StringIO(cfg_lines))
        else:
            self._read_io(cfg_lines)

    def _read_io(self, cfg):
        """Читает строки в формате CFG и сохраняет их данные."""
        line_count = 0
        self._nrates = 1
        self._sample_rates = []
        self._analog_channels = []
        self._status_channels = []

        # Первая строка
        line = cfg.readline()
        # информация о станции, устройстве и ревизии стандарта comtrade
        packed = _read_sep_values(line)
        if 3 == len(packed):
            # только ревизия 1999 года и выше имеет год ревизии стандарта
            self._station_name, self._rec_dev_id, self._rev_year = packed
            self._rev_year = self._rev_year.strip()

            if self._rev_year not in (REV_1991, REV_1999, REV_2013):
                if not self.ignore_warnings:
                    msg = WARNING_UNKNOWN_REVISION.format(self._rev_year)
                    warnings.warn(Warning(msg))
        elif 2 == len(packed):
            self._station_name, self._rec_dev_id = packed
            self._rev_year = REV_1991
        elif 3 < len(packed): # защита от запятых в имени и некорректного разделения
            self._station_name, self._rec_dev_id, self._rev_year = packed[0], ",".join(packed[1:-1]), packed[-1]
            self._rev_year = self._rev_year.strip()

            if self._rev_year not in (REV_1991, REV_1999, REV_2013):
                if not self.ignore_warnings:
                    msg = WARNING_UNKNOWN_REVISION.format(self._rev_year)
                    warnings.warn(Warning(msg))
                    self._rev_year = REV_1991
        else: # защита от пустого/некорректного имени
            self._station_name, self._rec_dev_id = packed[0], ""
            if not self.ignore_warnings:
                msg = WARNING_UNKNOWN_REVISION.format(self._rev_year)
                warnings.warn(Warning(msg))
                self._rev_year = REV_1991
                    
        line_count = line_count + 1

        # Вторая строка
        line = cfg.readline()
        # количество каналов и их тип
        totchn, achn, schn = _read_sep_values(line, 3, '0')
        self._channels_count = int(totchn)
        self._analog_count = int(achn[:-1])
        self._status_count = int(schn[:-1])
        self._analog_channels = [None]*self._analog_count
        self._status_channels = [None]*self._status_count
        line_count = line_count + 1

        # Строки описания аналоговых каналов
        for ichn in range(self._analog_count):
            line = cfg.readline()
            packed = _read_sep_values(line, 13, '0')
            # распаковать значения
            n, name, ph, ccbm, uu, a, b, skew, cmin, cmax, \
                primary, secondary, pors = packed
            # преобразование типов
            n = int(n)
            a = float(a)
            b = _prevent_null(b, float, 0.0)
            skew = _prevent_null(skew, float, 0.0)
            cmin = float(cmin)
            cmax = float(cmax)
            primary = float(primary)
            secondary = float(secondary)
            self.analog_channels[ichn] = AnalogChannel(n, a, b, skew, 
                cmin, cmax, name, uu, ph, ccbm, primary, secondary, pors)
            line_count = line_count + 1

        # Строки описания дискретных каналов
        for ichn in range(self._status_count):
            line = cfg.readline()
            # распаковать значения
            packed = _read_sep_values(line, 5, '0')
            n, name, ph, ccbm, y = packed
            # преобразование типов
            n = int(n)
            y = _prevent_null(y, int, 0)  # TODO: на самом деле критически важные данные. В будущем добавить предупреждение.
            self.status_channels[ichn] = StatusChannel(n, name, ph, ccbm, y)
            line_count = line_count + 1

        # Строка частоты
        line = cfg.readline()
        if len(line.strip()) > 0:
            self._frequency = float(line.strip())
        line_count = line_count + 1

        # Строка Nrates
        # количество различных частот дискретизации
        line = cfg.readline()
        self._nrates = int(line.strip())
        if self._nrates == 0:
            self._nrates = 1
            self._timestamp_critical = True
        else:
            self._timestamp_critical = False
        line_count = line_count + 1

        for inrate in range(self._nrates):
            line = cfg.readline()
            # каждая частота дискретизации
            samp, endsamp = _read_sep_values(line)
            samp = float(samp)
            endsamp = int(endsamp)
            self.sample_rates.append([samp, endsamp])
            line_count = line_count + 1

        # Время первой точки данных и временная база
        line = cfg.readline()
        ts_str = line.strip()
        self._start_timestamp, nanosec = _read_timestamp(
            ts_str,
            self.rev_year,
            self.ignore_warnings
        )
        self._time_base = self._get_time_base(nanosec)
        line_count = line_count + 1

        # Точка данных о событии и временная база
        line = cfg.readline()
        ts_str = line.strip()
        self._trigger_timestamp, nanosec = _read_timestamp(
            ts_str,
            self.rev_year,
            self.ignore_warnings
        )
        self._time_base = min([self.time_base, self._get_time_base(nanosec)])
        line_count = line_count + 1

        # Тип файла DAT
        line = cfg.readline()
        self._ft = line.strip()
        line_count = line_count + 1

        # Множитель временной метки
        if self._rev_year in (REV_1999, REV_2013):
            line = cfg.readline().strip()
            if len(line) > 0:
                self._time_multiplier = float(line)
            else:
                self._time_multiplier = 1.0
            line_count = line_count + 1

        # time_code и local_code
        if self._rev_year == REV_2013:
            line = cfg.readline()

            if line:
                self._time_code, self._local_code = _read_sep_values(line)
                line_count = line_count + 1

                line = cfg.readline()
                # time_code и local_code
                self._tmq_code, self._leap_second = _read_sep_values(line)
                line_count = line_count + 1

    def _get_time_base(self, using_nanoseconds: bool):
        """
        Возвращает временную базу, которая основана на дробной части
        секунд во временной метке (00.XXXXX).
        """
        if using_nanoseconds:
            return self.TIME_BASE_NANOSEC
        else:
            return self.TIME_BASE_MICROSEC
        
    def to_string(self):
        lines = []
        lines.append(f"{self._station_name},{self._rec_dev_id},{self._rev_year}")
        lines.append(f"{self._channels_count},{self._analog_count}A,{self._status_count}D")

        for ch in self._analog_channels:
            lines.append(f"{ch.index},{ch.name},{ch.phase},{ch.circuit},{ch.unit},{ch.multiplier},{ch.offset},{ch.skew},{ch.min_value},{ch.max_value},{ch.primary},{ch.secondary},{ch.ps}")

        for ch in self._status_channels:
            lines.append(f"{ch.index},{ch.name},{ch.phase},{ch.circuit},{ch.y}")

        lines.append(f"{self._frequency}")
        lines.append(f"{self._nrates}")
        for rate in self._sample_rates:
            samples = rate[0]
            duration = rate[1]
            lines.append(f"{samples},{duration}")

        lines.append(f"{self._start_timestamp.strftime('%d/%m/%Y,%H:%M:%S.%f')}")
        lines.append(f"{self._trigger_timestamp.strftime('%d/%m/%Y,%H:%M:%S.%f')}")
        lines.append(f"{self._ft}")
        lines.append(f"{self._time_multiplier}")

        return "\n".join(lines) + "\n"


class Comtrade:
    """Разбирает и хранит данные Comtrade."""
    # расширения
    EXT_CFG = "cfg"
    EXT_DAT = "dat"
    EXT_INF = "inf"
    EXT_HDR = "hdr"
    # специфика формата
    ASCII_SEPARATOR = ","
    
    def __init__(self, **kwargs):
        """
        Конструктор объекта Comtrade.

        Аргументы ключевого слова:
        ignore_warnings -- выводить ли предупреждения в stdout
            (по умолчанию: False).
        """
        self.file_path = ""

        self._cfg = Cfg(**kwargs)

        # Данные CFG по умолчанию
        self._analog_channel_ids = []
        self._analog_phases = []
        self._status_channel_ids = []
        self._status_phases = []
        self._timestamp_critical = False

        # Типы данных
        if "use_numpy_arrays" in kwargs:
            self._use_numpy_arrays = kwargs["use_numpy_arrays"]
        else:
            self._use_numpy_arrays = False

        # Данные файла DAT
        self._time_values = _preallocate_values("f", 0, self._use_numpy_arrays)
        self._analog_values = []
        self._status_values = []

        # Дополнительные данные CFF (или дополнительные файлы comtrade)
        self._hdr = None
        self._inf = None

        if "ignore_warnings" in kwargs:
            self.ignore_warnings = kwargs["ignore_warnings"]
        else:
            self.ignore_warnings = False

    @property
    def station_name(self) -> str:
        """Возвращает имя станции записывающего устройства."""
        return self._cfg.station_name

    @property
    def rec_dev_id(self) -> str:
        """Возвращает идентификатор записывающего устройства."""
        return self._cfg.rec_dev_id

    @property
    def rev_year(self) -> int:
        """Возвращает год ревизии COMTRADE."""
        return self._cfg.rev_year

    @property
    def cfg(self) -> Cfg:
        """Возвращает базовый экземпляр класса CFG."""
        return self._cfg

    @property
    def hdr(self):
        """Возвращает содержимое файла HDR."""
        return self._hdr
    
    @property
    def inf(self):
        """Возвращает содержимое файла INF."""
        return self._inf

    @property
    def analog_channel_ids(self) -> list:
        """Возвращает список имен аналоговых каналов."""
        return self._analog_channel_ids
    
    @property
    def analog_phases(self) -> list:
        """Возвращает список имен фаз аналоговых каналов."""
        return self._analog_phases
    
    @property
    def status_channel_ids(self) -> list:
        """Возвращает список имен дискретных каналов."""
        return self._status_channel_ids
    
    @property
    def status_phases(self) -> list:
        """Возвращает список имен фаз дискретных каналов."""
        return self._status_phases

    @property
    def time(self) -> list:
        """Возвращает список значений времени."""
        return self._time_values

    @property
    def analog(self) -> list:
        """Возвращает двумерный список значений аналоговых каналов."""
        return self._analog_values
    
    @property
    def status(self) -> list:
        """Возвращает двумерный список значений дискретных каналов."""
        return self._status_values

    @property
    def total_samples(self) -> int:
        """Возвращает общее количество выборок (на канал)."""
        return self._total_samples

    @property
    def frequency(self) -> float:
        """Возвращает измеренную частоту сети в герцах."""
        return self._cfg.frequency

    @property
    def start_timestamp(self):
        """Возвращает временную метку начала записи в виде объекта datetime."""
        return self._cfg.start_timestamp

    @property
    def trigger_timestamp(self):
        """Возвращает временную метку триггера в виде объекта datetime."""
        return self._cfg.trigger_timestamp

    @property
    def channels_count(self) -> int:
        """Возвращает общее количество каналов."""
        return self._cfg.channels_count
    
    @channels_count.setter
    def channels_count(self, count_corrector) -> None:
        """Устанавливает общее количество каналов."""
        self._cfg.channels_count = count_corrector

    @property
    def analog_count(self) -> int:
        """Возвращает количество аналоговых каналов."""
        return self._cfg.analog_count

    @analog_count.setter
    def analog_count(self, count_corrector) -> None:
        """Устанавливает количество аналоговых каналов."""
        self._cfg.analog_count = count_corrector

    @property
    def status_count(self) -> int:
        """Возвращает количество дискретных каналов."""
        return self._cfg.status_count

    @property
    def trigger_time(self) -> float:
        """Возвращает относительное время срабатывания в секундах."""
        stt = self._cfg.start_timestamp
        trg = self._cfg.trigger_timestamp
        tdiff = trg - stt
        tsec = (tdiff.days*60*60*24) + tdiff.seconds + (tdiff.microseconds*1E-6)
        return tsec

    @property
    def time_base(self) -> float:
        """Возвращает временную базу."""
        return self._cfg.time_base

    @property
    def ft(self) -> str:
        """Возвращает ожидаемый формат файла DAT."""
        return self._cfg.ft

    # Устаревшие свойства - заменено "Digital" на "Status"
    @property
    def digital_channel_ids(self) -> list:
        """Возвращает список имен дискретных каналов."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_channel_ids устарело, используйте status_channel_ids вместо него."))
        return self._status_channel_ids

    @property
    def digital(self) -> list:
        """Возвращает двумерный список значений дискретных каналов."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital устарело, используйте status вместо него."))
        return self._status_values
    
    @digital.setter
    def digital(self, new_digital_signals):
        self._digital = new_digital_signals

    @property
    def digital_count(self) -> int:
        """Возвращает количество дискретных каналов."""
        if not self.ignore_warnings:
            warnings.warn(FutureWarning("digital_count устарело, используйте status_count вместо него."))
        return self._cfg.status_count
    
    @digital_count.setter
    def digital_count(self, count_corrector) -> None:
        """Устанавливает количество дискретных каналов."""
        self._cfg.status_count = count_corrector

    def _get_dat_reader(self):
        # сравнение формата файла без учета регистра
        dat = None
        ft_upper = self.ft.upper()
        dat_kwargs = {"use_numpy_arrays": self._use_numpy_arrays}
        if ft_upper == TYPE_ASCII:
            dat = AsciiDatReader(**dat_kwargs)
        elif ft_upper == TYPE_BINARY:
            dat = BinaryDatReader(**dat_kwargs)
        elif ft_upper == TYPE_BINARY32:
            dat = Binary32DatReader(**dat_kwargs)
        elif ft_upper == TYPE_FLOAT32:
            dat = Float32DatReader(**dat_kwargs)
        else:
            dat = None
            raise Exception("Неподдерживаемый формат файла данных: {}".format(self.ft))
        return dat

    def read(self, cfg_lines, dat_lines_or_bytes) -> None:
        """
        Читает содержимое файлов CFG и DAT. Ожидает объекты FileIO или StringIO.
        """
        self._cfg.read(cfg_lines)

        # идентификаторы каналов
        self._cfg_extract_channels_ids(self._cfg)
        
        # фазы каналов
        self._cfg_extract_phases(self._cfg)

        dat = self._get_dat_reader()
        dat.read(dat_lines_or_bytes, self._cfg)

        # копирование информации об объекте dat
        self._dat_extract_data(dat)

    def _cfg_extract_channels_ids(self, cfg) -> None:
        self._analog_channel_ids = [channel.name for channel in cfg.analog_channels]
        self._status_channel_ids = [channel.name for channel in cfg.status_channels]
        
    def _cfg_extract_phases(self, cfg) -> None:
        self._analog_phases = [channel.ph for channel in cfg.analog_channels]
        self._status_phases = [channel.ph for channel in cfg.status_channels]

    def _dat_extract_data(self, dat) -> None:
        self._time_values   = dat.time
        self._analog_values = dat.analog
        self._status_values = dat.status
        self._total_samples = dat.total_samples

    def load(self, cfg_file, dat_file = None, **kwargs) -> None:
        """
        Загружает файлы CFG, DAT, INF и HDR. Каждый должен быть объектом FileIO или StringIO
        . dat_file, inf_file и hdr_file являются необязательными (по умолчанию: None).

        cfg_file - это путь к файлу cfg, включая его расширение.
        dat_file является необязательным и может быть установлен, если имя файла DAT отличается от
        имени файла CFG.

        Аргументы ключевого слова:
        inf_file -- необязательный путь к файлу INF (по умолчанию = None)
        hdr_file -- необязательный путь к файлу HDR (по умолчанию = None)
        """
        if "inf_file" in kwargs:
            inf_file = kwargs["inf_file"]
        else:
            inf_file = None

        if "hdr_file" in kwargs:
            hdr_file = kwargs["hdr_file"]
        else:
            hdr_file = None

        # какое расширение: CFG или CFF?
        file_ext = cfg_file[-3:].upper()
        if file_ext == "CFG":
            basename = cfg_file[:-3]
            # если не указано, выводим dat_file с cfg_file
            if dat_file is None:
                dat_file = cfg_file[:-3] + self.EXT_DAT

            if inf_file is None:
                inf_file = basename + self.EXT_INF

            if hdr_file is None:
                hdr_file = basename + self.EXT_HDR

            # загружаем и cfg, и dat
            file_kwargs = {}
            if "encoding" in kwargs:
                file_kwargs["encoding"] = kwargs["encoding"]
            self._load_cfg_dat(cfg_file, dat_file, **file_kwargs)

            # Загружаем дополнительные файлы inf и hdr, если они существуют.
            self._load_inf(inf_file, **file_kwargs)
            self._load_hdr(hdr_file, **file_kwargs)

        elif file_ext == "CFF":
            # проверяем, существует ли файл CFF
            self._load_cff(cfg_file)
        else:
            raise Exception(r"Ожидался путь к файлу CFG, вместо этого получено \"{}\".".format(cfg_file))

    def _load_cfg_dat(self, cfg_filepath, dat_filepath, **kwargs):
        self._cfg.load(cfg_filepath, **kwargs)

        # идентификаторы каналов
        self._cfg_extract_channels_ids(self._cfg)
        
        # фазы каналов
        self._cfg_extract_phases(self._cfg)

        dat = self._get_dat_reader()
        dat.load(dat_filepath, self._cfg, **kwargs)

        # копирование информации об объекте dat
        self._dat_extract_data(dat)

    def _load_inf(self, inf_file, **kwargs):
        if os.path.exists(inf_file):
            if "encoding" not in kwargs and _file_is_utf8(self.file_path):
                kwargs["encoding"] = "utf-8"
            with open(inf_file, 'r', **kwargs) as file:
                self._inf = file.read()
                if len(self._inf) == 0:
                    self._inf = None
        else:
            self._inf = None

    def _load_hdr(self, hdr_file, **kwargs):
        if os.path.exists(hdr_file):
            if "encoding" not in kwargs and _file_is_utf8(self.file_path):
                kwargs["encoding"] = "utf-8"
            with open(hdr_file, 'r', **kwargs) as file:
                self._hdr = file.read()
                if len(self._hdr) == 0:
                    self._hdr = None
        else:
            self._hdr = None

    def _load_cff(self, cff_file_path: str, **kwargs):
        # хранит строки каждого типа файла
        cfg_lines = []
        dat_lines = []
        hdr_lines = []
        inf_lines = []
        # тип файла: CFG, HDR, INF, DAT
        ftype = None
        # формат файла: ASCII, BINARY, BINARY32, FLOAT32
        fformat = None
        if "encoding" not in kwargs and _file_is_utf8(cff_file_path):
            kwargs["encoding"] = "utf-8"
        # Количество байтов для двоичных/float-данных
        fbytes = 0
        with open(cff_file_path, "r", **kwargs) as file:
            header_re = re.compile(CFF_HEADER_REXP)
            last_match = None
            line_number = 0
            line = file.readline()
            while line != "":
                line_number += 1
                mobj = header_re.match(line.strip().upper())
                if mobj is not None:
                    last_match = mobj
                    groups = last_match.groups()
                    ftype   = groups[0]
                    if len(groups) > 1:
                        fformat = last_match.groups()[1]
                        fbytes_obj = last_match.groups()[2]
                        fbytes  = int(fbytes_obj) if fbytes_obj is not None else 0

                elif last_match is not None and ftype == "CFG":
                    cfg_lines.append(line.strip())

                elif last_match is not None and ftype == "DAT":
                    if fformat == TYPE_ASCII:
                        dat_lines.append(line.strip())
                    else:
                        break

                elif last_match is not None and ftype == "HDR":
                    hdr_lines.append(line.strip())

                elif last_match is not None and ftype == "INF":
                    inf_lines.append(line.strip())

                line = file.readline()

        if fformat == TYPE_ASCII:
            # обработка данных ASCII CFF
            self.read("\n".join(cfg_lines), "\n".join(dat_lines))
        else:
            # чтение байтов dat
            total_bytes = os.path.getsize(cff_file_path)
            cff_bytes_read = total_bytes - fbytes
            with open(cff_file_path, "rb") as file:
                file.read(cff_bytes_read)
                dat_bytes = file.read(fbytes)
            self.read("\n".join(cfg_lines), dat_bytes)

        # хранит дополнительные данные
        self._hdr = "\n".join(hdr_lines)
        if len(self._hdr) == 0:
            self._hdr = None

        self._inf = "\n".join(inf_lines)
        if len(self._inf) == 0:
            self._inf = None

    def cfg_summary(self):
        """Возвращает строку с краткой информацией об атрибутах CFG."""
        header_line = "Каналы (всего,А,Д): {}A + {}D = {}"
        sample_line = "Частота дискретизации {} Гц до выборки #{}"
        interval_line = "От {} до {} с множителем времени = {}"
        format_line = "{} формат"

        lines = [header_line.format(self.analog_count, self.status_count,
                                    self.channels_count),
                 "Частота сети: {} Гц".format(self.frequency)]
        for i in range(self._cfg.nrates):
            rate, points = self._cfg.sample_rates[i]
            lines.append(sample_line.format(rate, points))
        lines.append(interval_line.format(self.start_timestamp,
                                          self.trigger_timestamp,
                                          self._cfg.timemult))
        lines.append(format_line.format(self.ft))
        return "\n".join(lines)

    def to_dataframe(self):
        """
        Преобразует загруженные данные осциллограммы в pandas DataFrame.
        
        Возвращает:
            pandas.DataFrame: DataFrame с временной колонкой и колонками для каждого
                              аналогового и дискретного канала.
        """
        if self.total_samples == 0:
            return pd.DataFrame() # Возвращаем пустой DataFrame, если нет данных

        # Создаем словарь для будущего DataFrame
        data = {'Time': self.time}

        # Добавляем аналоговые каналы
        for i, channel_name in enumerate(self.analog_channel_ids):
            # Проверяем, чтобы не было дубликатов имен столбцов
            if channel_name in data:
                # Если имя уже есть, добавляем суффикс
                unique_name = f"{channel_name}_{i}"
                data[unique_name] = self.analog[i]
            else:
                data[channel_name] = self.analog[i]

        # Добавляем дискретные каналы
        for i, channel_name in enumerate(self.status_channel_ids):
             # Проверяем, чтобы не было дубликатов имен столбцов
            if channel_name in data:
                unique_name = f"{channel_name}_status_{i}"
                data[unique_name] = self.status[i]
            else:
                data[channel_name] = self.status[i]

        return pd.DataFrame(data)

    def remove_disallowed_analog_names(self, allowed_names: dict) -> 'Comtrade':
        k = 0
        # TODO: Проверить корректность работы
        for i in range(len(self.analog)):
            if not (self.analog_channel_ids[i-k] in allowed_names):
                self.analog.pop(i-k)
                self.analog_channel_ids.pop(i-k)
                self.analog_phases.pop(i-k)
                self.analog_count -= 1
                self.channels_count -= 1
                k = k + 1
        
        return self
    
    def remove_disallowed_digital_names(self, allowed_names: dict) -> 'Comtrade':
        k = 0
        # TODO: Проверить корректность работы
        for i in range(len(self.digital)):
            if not (self.digital_channel_ids[i-k] in allowed_names):
                self.digital.pop(i-k)
                self.digital_channel_ids.pop(i-k)
                self.status_phases.pop(i-k)
                self.digital_count -= 1
                self.channels_count -= 1
                k = k + 1
        
        return self

    def write_cfg(self, filepath, cfg):
        with open(filepath, 'w') as file:
            file.write(cfg.to_string())  # Предполагается, что у вас есть метод для генерации содержимого файла конфигурации

    def write_to_file(self, cfg_filepath, dat_filepath):
        # Запись файла конфигурации
        self.write_cfg(cfg_filepath, self._cfg)
        
        
        # FIXME: !! Запись в dat-файл не реализовано и не работает!!
        # Подготовка данных для записи
        # data_to_write = []
        # for i in range(self.total_samples):
        #     time = self.time[i]
        #     sample_number = i
        #     analog = [self.analog[j][i] * self._cfg.analog_channels[j].multiplier for j in range(self.analog_count)]
        #     status = [int(self.status[j][i]) for j in range(self.status_count)]
        #     data_to_write.append((time, sample_number, analog, status))
        # # Запись файла данных в соответствующем формате
        # self._dat_writer = Float32DatWriter(dat_filepath, self.analog_channel_ids, self.digital_channel_ids)
        # self._dat_writer.write_data(dat_filepath, data_to_write)

class Channel:
    """Хранит общие данные описания канала."""
    def __init__(self, n=1, name='', ph='', ccbm=''):
        """Абстрактный конструктор класса Channel."""
        self.n = n
        self.name = name
        self.ph = ph
        self.ccbm = ccbm

    def __str__(self):
        return ','.join([str(self.n), self.name, self.ph, self.ccbm])


class StatusChannel(Channel):
    """Хранит данные описания дискретного канала."""
    def __init__(self, index: int, name='', phase='', circuit='', y=0):
        """Конструктор класса StatusChannel."""
        super().__init__(index, name, phase, circuit)
        self.name = name
        self.index = index
        self.name = name
        self.phase = phase
        self.circuit = circuit
        self.y = y

    def __str__(self):
        fields = [str(self.index), self.name, self.phase, self.circuit, str(self.y)]


class AnalogChannel(Channel):
    """Хранит данные описания аналогового канала."""
    def __init__(self, index: int, multiplier: float, offset=0.0, skew=0.0, min_value=-32767,
                 max_value=32767, name='', unit='', phase='', circuit='', primary=1.0,
                 secondary=1.0, ps='P'):
        """Конструктор класса AnalogChannel."""
        super().__init__(index, name, phase, circuit)
        self.index = index
        self.name = name
        self.unit = unit
        self.multiplier = multiplier
        self.offset = offset
        self.skew = skew
        self.min_value = min_value
        self.max_value = max_value
        # разное
        self.unit = unit
        self.phase = phase
        self.circuit = circuit
        self.primary = primary
        self.secondary = secondary
        self.ps = ps

    def __str__(self):
        fields = [str(self.index), self.name, self.phase, self.circuit, self.unit, 
            str(self.multiplier), str(self.offset), str(self.skew), str(self.min_value), 
            str(self.max_value), str(self.primary), str(self.secondary), self.ps]
        return ','.join(fields)


class DatReader:
    """Абстрактный класс DatReader. Используется для разбора содержимого DAT-файла."""
    read_mode = "r"

    def __init__(self, **kwargs):
        """Конструктор класса DatReader."""
        if "use_numpy_arrays" in kwargs:
            self._use_numpy_arrays = kwargs["use_numpy_arrays"]
        else:
            self._use_numpy_arrays = False
        self.file_path = ""
        self._content = None
        self._cfg = None
        self.time = _preallocate_values("f", 0, self._use_numpy_arrays)
        self.analog = []
        self.status = []
        self._total_samples = 0

    @property
    def total_samples(self):
        """Возвращает общее количество выборок (на канал)."""
        return self._total_samples

    def load(self, dat_filepath, cfg, **kwargs):
        """Загружает DAT-файл и разбирает его содержимое."""
        self.file_path = dat_filepath
        self._content = None
        if os.path.isfile(self.file_path):
            # извлечение информации о размерах данных из файла CFG
            self._cfg = cfg
            self._preallocate()
            if "encoding" not in kwargs and self.read_mode != "rb" and \
                    _file_is_utf8(self.file_path):
                kwargs["encoding"] = "utf-8"
            with open(self.file_path, self.read_mode, **kwargs) as contents:
                self.parse(contents)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self.file_path)

    def read(self, dat_lines, cfg):
        """
        Читает содержимое DAT-файла, ожидая список строк или объект FileIO.
        """
        self.file_path = None
        self._content = dat_lines
        self._cfg = cfg
        self._preallocate()
        self.parse(dat_lines)

    def _preallocate(self):
        # чтение из файла cfg количества выборок в файле dat
        steps = self._cfg.sample_rates[-1][1]  # последнее поле samp
        self._total_samples = steps

        # количество аналоговых и дискретных каналов
        analog_count = self._cfg.analog_count
        status_count = self._cfg.status_count

        # предварительное выделение памяти для аналоговых и дискретных значений
        self.time = _preallocate_values("f", steps, self._use_numpy_arrays)
        self.analog = [None] * analog_count
        self.status = [None] * status_count
        # предварительное выделение памяти для значений каждого канала нулями
        for i in range(analog_count):
            self.analog[i] = _preallocate_values("f", steps,
                self._use_numpy_arrays)
        for i in range(status_count):
            self.status[i] = _preallocate_values("i", steps,
                self._use_numpy_arrays)

    def _get_samp(self, n) -> float:
        """Получает частоту дискретизации для выборки n (индекс с 1)."""
        # TODO: сделать тесты.
        last_sample_rate = 1.0
        for samp, endsamp in self._cfg.sample_rates:
            if n <= endsamp:
                return samp
        return last_sample_rate

    def _get_time(self, n: int, ts_value: float, time_base: float,
                  time_multiplier: float):
        # TODO: добавить опцию для принудительного использования временной метки из dat-файла, когда она доступна.
        # TODO: сделать тесты.
        ts = 0
        sample_rate = self._get_samp(n)
        if not self._cfg.timestamp_critical or ts_value == TIMESTAMP_MISSING:
            # если временная метка отсутствует, использовать расчетную.
            if sample_rate != 0.0:
                return (n - 1) / sample_rate
            else:
                raise Exception("Отсутствует временная метка и не указана частота дискретизации"
                                "provided.")
        else:
            # Использовать предоставленную временную метку, если она не отсутствует
            return ts_value * time_base * time_multiplier

    def parse(self, contents):
        """Виртуальный метод, разбирает содержимое DAT-файла."""
        pass


class AsciiDatReader(DatReader):
    """Подкласс DatReader для формата ASCII."""
    def __init__(self, **kwargs):
        # Вызов инициализации для унаследованного класса
        super().__init__(**kwargs)
        self.ASCII_SEPARATOR = SEPARATOR

        self.DATA_MISSING = ""

    def parse(self, contents):
        """Разбирает содержимое файла ASCII."""
        analog_count = self._cfg.analog_count
        status_count = self._cfg.status_count
        time_mult = self._cfg.timemult
        time_base = self._cfg.time_base

        # вспомогательные векторы (усиления и смещения каналов)
        a = [x.multiplier for x in self._cfg.analog_channels]
        b = [x.offset for x in self._cfg.analog_channels]

        # извлечь строки
        if type(contents) is str:
            lines = contents.splitlines()
        else:
            lines = contents

        line_number = 0
        for line in lines:
            line_number = line_number + 1
            if line_number > self._total_samples:
                break
            values = line.strip().split(self.ASCII_SEPARATOR)

            n = int(values[0])
            # Чтение времени
            ts_val = float(values[1])
            ts = self._get_time(n, ts_val, time_base, time_mult)

            avalues = [float(x)*a[i] + b[i] for i, x in enumerate(values[2:analog_count+2])]
            svalues = [int(x) for x in values[len(values)-status_count:]]

            # сохранить
            self.time[line_number-1] = ts
            for i in range(analog_count):
                self.analog[i][line_number - 1] = avalues[i]
            for i in range(status_count):
                self.status[i][line_number - 1] = svalues[i]

class BinaryDatReader(DatReader):
    """Подкласс DatReader для 16-битного двоичного формата."""
    def __init__(self, **kwargs):
        # Вызов инициализации для унаследованного класса
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 2
        self.STATUS_BYTES = 2
        self.TIME_BYTES = 4
        self.SAMPLE_NUMBER_BYTES = 4

        # максимальное отрицательное значение
        self.DATA_MISSING = 0xFFFF

        self.read_mode = "rb"

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}h {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}h"
            self.STRUCT_FORMAT_STATUS_ONLY = "LL {dcount:d}H"
        else:
            self.STRUCT_FORMAT = "II {acount:d}h {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}h"
            self.STRUCT_FORMAT_STATUS_ONLY = "II {dcount:d}H"

    def get_reader_format(self, analog_channels, status_bytes):
        # Количество полей состояния размером 2 байта на основе общего количества
        # байтов.
        dcount = math.floor(status_bytes / 2)
        
        # Проверка конфигурации файла
        if int(status_bytes) > 0 and int(analog_channels) > 0:
            return self.STRUCT_FORMAT.format(acount=analog_channels, 
                dcount=dcount)
        elif int(analog_channels) > 0:
            # Только аналоговые каналы.
            return self.STRUCT_FORMAT_ANALOG_ONLY.format(acount=analog_channels)
        else:
            # Только дискретные каналы.
            return self.STRUCT_FORMAT_STATUS_ONLY.format(acount=dcount)

    def parse(self, contents):
        """Разбирает содержимое двоичного файла DAT."""
        time_mult = self._cfg.timemult
        time_base = self._cfg.time_base
        achannels = self._cfg.analog_count
        schannel = self._cfg.status_count

        # вспомогательные векторы (усиления и смещения каналов)
        a = [x.multiplier for x in self._cfg.analog_channels]
        b = [x.offset for x in self._cfg.analog_channels]

        sample_id_bytes = self.SAMPLE_NUMBER_BYTES + self.TIME_BYTES
        abytes = achannels*self.ANALOG_BYTES
        dbytes = self.STATUS_BYTES * math.ceil(schannel / 16.0)
        bytes_per_row = sample_id_bytes + abytes + dbytes
        groups_of_16bits = math.floor(dbytes / self.STATUS_BYTES)

        # Формат структуры.
        row_reader = struct.Struct(self.get_reader_format(achannels, dbytes))

        # Функция чтения строк.
        next_row = None
        if isinstance(contents, io.TextIOBase) or \
                isinstance(contents, io.BufferedIOBase):
            # Чтение всего содержимого буфера
            contents = contents.read()

        for irow, values in enumerate(row_reader.iter_unpack(contents)):
            # Номер выборки
            n = values[0]
            # Временная метка
            ts_val = values[1]
            ts = self._get_time(n, ts_val, time_base, time_mult)

            if irow >= self.total_samples:
                break
            self.time[irow] = ts

            # Извлечение значений аналоговых каналов.
            for ichannel in range(achannels):
                yint = values[ichannel + 2]
                y = a[ichannel] * yint + b[ichannel]
                self.analog[ichannel][irow] = y

            # Извлечение значений дискретных каналов.
            for igroup in range(groups_of_16bits):
                group = values[achannels + 2 + igroup]

                # для каждой группы из 16 бит извлекаем дискретные каналы
                maxchn = min([ (igroup+1) * 16, schannel])
                for ichannel in range(igroup * 16, maxchn):
                    chnindex = ichannel - igroup*16
                    mask = int('0b01', 2) << chnindex
                    extract = (group & mask) >> chnindex

                    self.status[ichannel][irow] = extract

            # Получение следующей строки
            irow += 1


class Binary32DatReader(BinaryDatReader):
    """Подкласс DatReader для 32-битного двоичного формата."""
    def __init__(self, **kwargs):
        # Вызов инициализации для унаследованного класса
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 4

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}l {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}l"
        else:
            self.STRUCT_FORMAT = "II {acount:d}i {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}i"

        # максимальное отрицательное значение
        self.DATA_MISSING = 0xFFFFFFFF


class Float32DatReader(BinaryDatReader):
    """Подкласс DatReader для двоичного формата с плавающей запятой одинарной точности."""
    def __init__(self, **kwargs):
        # Вызов инициализации для унаследованного класса
        super().__init__(**kwargs)
        self.ANALOG_BYTES = 4

        if struct.calcsize("L") == 4:
            self.STRUCT_FORMAT = "LL {acount:d}f {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "LL {acount:d}f"
        else:
            self.STRUCT_FORMAT = "II {acount:d}f {dcount:d}H"
            self.STRUCT_FORMAT_ANALOG_ONLY = "II {acount:d}f"

        # Максимальное отрицательное значение
        self.DATA_MISSING = sys.float_info.min
