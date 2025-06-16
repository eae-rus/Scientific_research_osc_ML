import os
import hashlib
import pandas as pd
from comtrade_APS import Comtrade # Assuming comtrade_APS.py is in PYTHONPATH
from dataflow.comtrade_processing import ReadComtrade # Assuming dataflow is in PYTHONPATH

class Oscillogram:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"COMTRADE file not found: {file_path}")

        self.filepath = file_path
        self._raw_comtrade: Comtrade = None
        self._df: pd.DataFrame = None

        # Attempt to determine .dat file path for hashing
        base_path, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        if ext_lower == '.cfg':
            self._dat_file_path = base_path + '.dat'
        elif ext_lower == '.cff':
            # For CFF, DAT path determination is complex.
            # The Comtrade loader handles embedded DAT or referenced DAT.
            # For hashing, if DAT is embedded, we might need to hash file_path itself,
            # or rely on loader to expose DAT bytes if possible.
            # For now, if it's a CFF, we might not have a separate .dat file.
            # The comtrade_APS.Comtrade().load() method should handle CFF loading.
            # Hashing will fallback if a distinct .dat file isn't identified here.
            self._dat_file_path = None # Or file_path if data is embedded and loader doesn't separate it
        elif ext_lower == '.dat':
            # If a .dat file is passed directly, use it.
            self._dat_file_path = file_path
        else:
            # Fallback for unknown extensions or if .cfg is expected by loader
            # but another extension is given.
            self._dat_file_path = None


        # Load the COMTRADE file
        try:
            # Using Comtrade directly from comtrade_APS
            self._raw_comtrade = Comtrade(ignore_warnings=True, use_numpy_arrays=True)
            self._raw_comtrade.load(cfg_file=self.filepath) # load should handle .cfg, .cff
            self._df = self._raw_comtrade.to_dataframe()
        except ImportError as ie:
            # Handle missing pandas/numpy specifically if to_dataframe raises it
            raise RuntimeError(f"Failed to load COMTRADE file {self.filepath} due to missing dependency: {ie}")
        except Exception as e:
            raise RuntimeError(f"Failed to load COMTRADE file {self.filepath}: {e}")

        # Metadata extraction
        self.station_name = self._raw_comtrade.station_name
        self.rec_dev_id = self._raw_comtrade.rec_dev_id
        self.rev_year = self._raw_comtrade.rev_year
        self.frequency = self._raw_comtrade.frequency
        self.start_timestamp = self._raw_comtrade.start_timestamp
        self.trigger_timestamp = self._raw_comtrade.trigger_timestamp
        self.total_samples = self._raw_comtrade.total_samples

        self._file_hash = self._calculate_file_hash()

    def _calculate_file_hash(self) -> str:
        # Try to hash the .dat file if its path is known and it exists
        # For CFF files, self._dat_file_path might be None or same as self.filepath
        # If DAT is embedded in CFF, hashing self.filepath might be more appropriate if loader doesn't split.
        # However, comtrade_APS's load for CFF reads DAT part separately if it's ASCII.
        # For binary CFF, it reads bytes. Hashing the original CFF might be an option too.

        hash_target_path = None
        if self._dat_file_path and os.path.exists(self._dat_file_path):
            # If a distinct .dat file path was identified and exists
            hash_target_path = self._dat_file_path
        elif os.path.splitext(self.filepath)[1].lower() == '.cff':
            # For CFF files, if no separate .dat was identified (e.g. embedded binary data),
            # consider hashing the CFF file itself as a representation of its content.
            hash_target_path = self.filepath

        if hash_target_path:
            try:
                with open(hash_target_path, 'rb') as f:
                    file_bytes = f.read()
                    return hashlib.md5(file_bytes).hexdigest()
            except IOError:
                # Fallback if .dat or .cff can't be read
                pass

        # Fallback: hash the base filename (without extension) as a last resort
        # This is not a content hash but provides some uniqueness.
        return hashlib.md5(os.path.splitext(os.path.basename(self.filepath))[0].encode('utf-8')).hexdigest()

    @property
    def file_hash(self) -> str:
        return self._file_hash

    @property
    def raw_comtrade_obj(self) -> Comtrade:
        return self._raw_comtrade

    @property
    def cfg(self): # Type hint could be comtrade_APS.Cfg
        if self._raw_comtrade:
            return self._raw_comtrade.cfg
        return None

    @property
    def data_frame(self) -> pd.DataFrame:
        return self._df

    @property
    def analog_channel_ids(self) -> list:
        if self._raw_comtrade:
            return self._raw_comtrade.analog_channel_ids
        return []

    @property
    def status_channel_ids(self) -> list:
        if self._raw_comtrade:
            return self._raw_comtrade.status_channel_ids
        return []

    @property
    def time_values(self) -> list: # Or np.ndarray if numpy is used
        if self._df is not None and 'time' in self._df.columns:
            return self._df['time'].tolist() # Or .values for numpy array
        elif self._raw_comtrade:
             # Fallback if DataFrame is not populated with time or direct access needed
            return self._raw_comtrade.time
        return []

    def get_analog_series(self, channel_id: str) -> pd.Series:
        if self._df is not None and channel_id in self._df.columns:
            return self._df[channel_id]
        # The fallback below might be redundant if _df is guaranteed by __init__
        elif self._raw_comtrade and channel_id in self._raw_comtrade.analog_channel_ids:
            idx = self._raw_comtrade.analog_channel_ids.index(channel_id)
            # Assuming self._raw_comtrade.time is available and matches length of analog data
            return pd.Series(data=self._raw_comtrade.analog[idx], index=self._raw_comtrade.time, name=channel_id)
        raise KeyError(f"Analog channel '{channel_id}' not found or data not loaded.")

    def get_status_series(self, channel_id: str) -> pd.Series:
        if self._df is not None and channel_id in self._df.columns:
            return self._df[channel_id]
        # The fallback below might be redundant
        elif self._raw_comtrade and channel_id in self._raw_comtrade.status_channel_ids:
            idx = self._raw_comtrade.status_channel_ids.index(channel_id)
            return pd.Series(data=self._raw_comtrade.status[idx], index=self._raw_comtrade.time, name=channel_id)
        raise KeyError(f"Status channel '{channel_id}' not found or data not loaded.")

    def __repr__(self):
        return f"<Oscillogram(filepath='{self.filepath}', station='{self.station_name}', hash='{self.file_hash[:8]}...')>"
