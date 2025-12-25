import pytest
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import io

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.core.comtrade_custom import (
    _read_sep_values,
    _preallocate_values,
    _prevent_null,
    _get_date,
    _get_time,
    fill_with_zeros_to_the_right,
    _read_timestamp,
    Cfg,
    Comtrade,
    AsciiDatReader,
    BinaryDatReader,
    Binary32DatReader,
    Float32DatReader,
    REV_1991,
    REV_1999,
    REV_2013,
    TYPE_ASCII
)

class TestComtradeHelpers:
    """Тесты для вспомогательных функций comtrade_custom."""

    def test_read_sep_values(self):
        """Тест парсинга значений через запятую."""
        line = "val1, val2 , val3"
        assert _read_sep_values(line) == ("val1", "val2", "val3")
        assert _read_sep_values(line, expected=2) == ["val1", "val2"]
        assert _read_sep_values(line, expected=4, default="def") == ["val1", "val2", "val3", "def"]

    def test_preallocate_values(self):
        """Тест предварительного выделения памяти для массивов."""
        # С numpy
        arr = _preallocate_values("f", 10, use_numpy_arrays=True)
        assert len(arr) == 10
        assert arr.dtype == np.float32
        
        arr_i = _preallocate_values("i", 5, use_numpy_arrays=True)
        assert len(arr_i) == 5
        assert arr_i.dtype == np.int32

    def test_prevent_null(self):
        """Тест обработки пустых строк."""
        assert _prevent_null("123", int, 0) == 123
        assert _prevent_null("  ", int, 99) == 99
        assert _prevent_null("", float, 0.0) == 0.0

    def test_get_date(self):
        """Тест парсинга даты."""
        assert _get_date("24/12/2025") == (24, 12, 2025)
        assert _get_date("1/1/99") == (1, 1, 99)
        assert _get_date("invalid") == (0, 0, 0)

    def test_get_time(self):
        """Тест парсинга времени."""
        # ЧЧ:ММ:СС.мммммм
        h, m, s, ms, nano = _get_time("12:30:45.123456")
        assert (h, m, s, ms) == (12, 30, 45, 123456)
        assert not nano
        
        # С наносекундами (будет предупреждение, если не игнорировать)
        with pytest.warns(Warning, match="наносекундным"):
            h, m, s, ms, nano = _get_time("12:30:45.123456789")
        assert nano
        assert ms == 123456 # Усекается до микросекунд

    def test_fill_with_zeros_to_the_right(self):
        """Тест дополнения нулями справа."""
        assert fill_with_zeros_to_the_right("123", 6) == "123000"
        assert fill_with_zeros_to_the_right("123456", 6) == "123456"
        assert fill_with_zeros_to_the_right("1234567", 6) == "1234567"

    def test_read_timestamp(self):
        """Тест чтения временной метки."""
        ts_line = "24/12/2025, 12:30:45.123456"
        ts, nano = _read_timestamp(ts_line, REV_2013)
        assert ts == dt.datetime(2025, 12, 24, 12, 30, 45, 123456)
        
        # Формат 1991 (ММ/ДД/ГГГГ)
        ts_line_91 = "12/24/2025, 12:30:45.123456"
        ts, nano = _read_timestamp(ts_line_91, REV_1991)
        assert ts == dt.datetime(2025, 12, 24, 12, 30, 45, 123456)

class TestCfgClass:
    """Тесты для класса Cfg."""

    def test_cfg_init(self):
        """Тест инициализации Cfg."""
        cfg = Cfg()
        assert cfg.station_name == ""
        assert cfg.rev_year == 2013
        assert cfg.ft == TYPE_ASCII

    def test_cfg_read_from_string(self):
        """Тест чтения CFG из строки."""
        # Формат аналогового канала: n, name, ph, ccbm, uu, a, b, skew, cmin, cmax, primary, secondary, pors
        cfg_content = """Station A, Device 1, 2013
3, 2A, 1D
1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
2, IB, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
1, DI1, , , 0
50.0
1
1600, 1000
24/12/2025, 12:00:00.000000
24/12/2025, 12:00:00.100000
ASCII
1.0
"""
        cfg = Cfg()
        cfg.read(cfg_content)
        
        assert cfg.station_name == "Station A"
        assert cfg.rec_dev_id == "Device 1"
        assert cfg.rev_year == "2013"
        assert cfg.channels_count == 3
        assert cfg.analog_count == 2
        assert cfg.status_count == 1
        assert cfg.frequency == 50.0
        assert cfg.nrates == 1
        assert cfg.sample_rates[0] == [1600.0, 1000]
        assert cfg.ft == "ASCII"

class TestComtradeClass:
    """Тесты для основного класса Comtrade."""

    def test_comtrade_init(self):
        """Тест инициализации Comtrade."""
        obj = Comtrade()
        assert obj.cfg is not None
        assert isinstance(obj.cfg, Cfg)

    @patch('osc_tools.core.comtrade_custom.Cfg.load')
    @patch('osc_tools.core.comtrade_custom.Comtrade._load_cfg_dat')
    @patch('osc_tools.core.comtrade_custom.Comtrade._load_inf')
    @patch('osc_tools.core.comtrade_custom.Comtrade._load_hdr')
    @patch('os.path.exists', return_value=True)
    def test_comtrade_load_cfg_only(self, mock_exists, mock_hdr, mock_inf, mock_load_cfg_dat, mock_cfg_load):
        """Тест загрузки только CFG."""
        obj = Comtrade()
        obj.load('test.cfg')
        mock_load_cfg_dat.assert_called_once()

class TestComtradeData:
    """Тесты для обработки данных в Comtrade."""

    def test_ascii_dat_reader(self):
        """Тест AsciiDatReader."""
        cfg = Cfg()
        # Формат аналогового канала: n, name, ph, ccbm, uu, a, b, skew, cmin, cmax, primary, secondary, pors
        cfg_content = """Station A, Device 1, 2013
3, 2A, 1D
1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
2, IB, , , A, 2.0, 10.0, 0.0, -1000, 1000, 1.0, 1.0, P
1, DI1, , , 0
50.0
1
1600, 2
24/12/2025, 12:00:00.000000
24/12/2025, 12:00:00.100000
ASCII
1.0
"""
        cfg.read(cfg_content)
        
        dat_content = """1, 0, 100, 50, 1
2, 625, 200, 100, 0"""
        
        reader = AsciiDatReader()
        reader._cfg = cfg
        reader._total_samples = 2
        reader._preallocate()
        reader.parse(dat_content)
        
        # Проверка аналоговых данных
        # IA: 100 * 1.0 + 0.0 = 100.0
        # IB: 50 * 2.0 + 10.0 = 110.0
        assert reader.analog[0][0] == 100.0
        assert reader.analog[1][0] == 110.0
        
        # Проверка дискретных данных
        assert reader.status[0][0] == 1
        assert reader.status[0][1] == 0
        
        # Проверка времени
        assert reader.time[0] == pytest.approx(0.0)
        assert reader.time[1] == pytest.approx(0.000625)

    def test_comtrade_to_dataframe(self):
        """Тест преобразования в DataFrame."""
        obj = Comtrade()
        obj._analog_channel_ids = ['IA', 'IB']
        obj._status_channel_ids = ['DI1']
        obj._time_values = [0.0, 625.0]
        obj._analog_values = [[100.0, 200.0], [110.0, 210.0]]
        obj._status_values = [[1, 0]]
        obj._total_samples = 2
        
        df = obj.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Time' in df.columns
        assert 'IA' in df.columns
        assert 'IB' in df.columns
        assert 'DI1' in df.columns
        assert df.iloc[0]['IA'] == 100.0
        assert df.iloc[1]['DI1'] == 0

    def test_binary_dat_reader(self):
        """Тест BinaryDatReader."""
        cfg = Cfg()
        cfg_content = """Station A, Device 1, 2013
3, 2A, 1D
1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
2, IB, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
1, DI1, , , 0
50.0
1
1600, 1
24/12/2025, 12:00:00.000000
24/12/2025, 12:00:00.100000
BINARY
1.0
"""
        cfg.read(cfg_content)
        
        import struct
        # 1 sample: n=1, ts=0, IA=100, IB=200, DI1=1
        # Format: II hh H (assuming 2 analogs, 1 status)
        dat_content = struct.pack("<IIhhH", 1, 0, 100, 200, 1)
        
        reader = BinaryDatReader()
        reader._cfg = cfg
        reader._total_samples = 1
        reader._preallocate()
        reader.parse(dat_content)
        
        assert reader.analog[0][0] == 100.0
        assert reader.analog[1][0] == 200.0
        assert reader.status[0][0] == 1
        assert reader.time[0] == 0.0

    def test_binary32_dat_reader(self):
        """Тест Binary32DatReader."""
        cfg = Cfg()
        cfg_content = """Station A, Device 1, 2013
3, 2A, 1D
1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
2, IB, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
1, DI1, , , 0
50.0
1
1600, 1
24/12/2025, 12:00:00.000000
24/12/2025, 12:00:00.100000
BINARY32
1.0
"""
        cfg.read(cfg_content)
        
        import struct
        # 1 sample: n=1, ts=0, IA=100000, IB=200000, DI1=1
        # Format: II ii H (assuming 2 analogs, 1 status)
        # Binary32 uses 'l' or 'i' for analog
        fmt = "IIiiH" if struct.calcsize("L") == 4 else "IIiiH"
        # Wait, Binary32DatReader uses "LL {acount:d}l {dcount:d}H" or "II {acount:d}i {dcount:d}H"
        # Let's check the code again.
        # if struct.calcsize("L") == 4: self.STRUCT_FORMAT = "LL {acount:d}l {dcount:d}H"
        # else: self.STRUCT_FORMAT = "II {acount:d}i {dcount:d}H"
        
        if struct.calcsize("L") == 4:
            dat_content = struct.pack("<LLllH", 1, 0, 100000, 200000, 1)
        else:
            dat_content = struct.pack("<IIiiH", 1, 0, 100000, 200000, 1)
        
        reader = Binary32DatReader()
        reader._cfg = cfg
        reader._total_samples = 1
        reader._preallocate()
        reader.parse(dat_content)
        
        assert reader.analog[0][0] == 100000.0
        assert reader.analog[1][0] == 200000.0
        assert reader.status[0][0] == 1
        assert reader.time[0] == 0.0

    def test_float32_dat_reader(self):
        """Тест Float32DatReader."""
        cfg = Cfg()
        cfg_content = """Station A, Device 1, 2013
3, 2A, 1D
1, IA, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
2, IB, , , A, 1.0, 0.0, 0.0, -1000, 1000, 1.0, 1.0, P
1, DI1, , , 0
50.0
1
1600, 1
24/12/2025, 12:00:00.000000
24/12/2025, 12:00:00.100000
FLOAT32
1.0
"""
        cfg.read(cfg_content)
        
        import struct
        # 1 sample: n=1, ts=0, IA=123.45, IB=678.90, DI1=1
        if struct.calcsize("L") == 4:
            dat_content = struct.pack("<LLffH", 1, 0, 123.45, 678.90, 1)
        else:
            dat_content = struct.pack("<IIffH", 1, 0, 123.45, 678.90, 1)
        
        reader = Float32DatReader()
        reader._cfg = cfg
        reader._total_samples = 1
        reader._preallocate()
        reader.parse(dat_content)
        
        assert reader.analog[0][0] == pytest.approx(123.45, rel=1e-5)
        assert reader.analog[1][0] == pytest.approx(678.90, rel=1e-5)
        assert reader.status[0][0] == 1
        assert reader.time[0] == 0.0

    def test_comtrade_save_ascii(self):
        """Тест сохранения Comtrade в формате ASCII."""
        obj = Comtrade()
        obj.cfg._station_name = "Test Station"
        obj.cfg._ft = "ASCII"
        obj._time_values = [0.0, 0.001]
        obj._analog_values = [[10.0, 20.0]]
        obj._status_values = [[1, 0]]
        obj._total_samples = 2
        
        # Настройка аналогового канала
        from osc_tools.core.comtrade_custom import AnalogChannel
        obj.cfg._analog_channels = [AnalogChannel(1, 1.0, 0.0, name='IA')]
        obj.cfg._analog_count = 1
        obj.cfg._status_channels = [MagicMock(index=1, name='DI1', y=0)]
        obj.cfg._status_count = 1
        
        # Мокаем open
        mock_file = MagicMock()
        with patch('builtins.open', return_value=mock_file) as mock_open:
            obj.save('test.cfg')
            
        # Проверяем, что open вызывался
        assert mock_open.called

    def test_comtrade_save_binary(self):
        """Тест сохранения Comtrade в формате BINARY."""
        obj = Comtrade()
        obj.cfg._ft = "BINARY"
        obj._time_values = [0.0]
        obj._analog_values = [[100.0]]
        obj._status_values = [[1]]
        obj._total_samples = 1
        
        from osc_tools.core.comtrade_custom import AnalogChannel
        obj.cfg._analog_channels = [AnalogChannel(1, 1.0, 0.0, name='IA')]
        obj.cfg._analog_count = 1
        obj.cfg._status_count = 1
        
        mock_file = MagicMock()
        with patch('builtins.open', return_value=mock_file) as mock_open:
            obj.save('test.cfg')
            
        assert mock_open.called

