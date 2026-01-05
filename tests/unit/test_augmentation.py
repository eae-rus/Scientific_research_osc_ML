import pytest
import torch
import numpy as np
import polars as pl
from osc_tools.ml.augmentation import TimeSeriesAugmenter
from osc_tools.ml.dataset import OscillogramDataset

class TestTimeSeriesAugmenter:
    
    @pytest.fixture
    def sample_data(self):
        # (Batch, Time, Channels)
        # 8 channels: IA, IB, IC, In, UA, UB, UC, Un
        batch = 2
        time = 100
        channels = 8
        return torch.ones(batch, time, channels)

    def test_inversion(self, sample_data):
        config = {"p_inversion": 1.0}
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        assert torch.allclose(output, -sample_data)

    def test_scaling(self, sample_data):
        config = {
            "p_scaling": 1.0,
            "scaling_range_current": (2.0, 2.0), # Fixed scaling
            "scaling_range_voltage": (0.5, 0.5)
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        # Currents (0, 1, 2, 3) should be doubled
        assert torch.allclose(output[:, :, :4], sample_data[:, :, :4] * 2.0)
        # Voltages (4, 5, 6, 7) should be halved
        assert torch.allclose(output[:, :, 4:], sample_data[:, :, 4:] * 0.5)

    def test_noise(self, sample_data):
        config = {
            "p_noise": 1.0,
            "noise_std_current": 0.1,
            "noise_std_voltage": 0.1
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        assert not torch.allclose(output, sample_data)
        # Check that mean is roughly same (noise is zero mean)
        assert torch.abs(output.mean() - sample_data.mean()) < 0.1

    def test_offset(self, sample_data):
        config = {
            "p_offset": 1.0,
            "offset_range": (1.0, 1.0)
        }
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        assert torch.allclose(output, sample_data + 1.0)

    def test_phase_shuffling(self):
        # Create distinct phases
        # IA=1, IB=2, IC=3
        # UA=10, UB=20, UC=30
        data = torch.zeros(1, 10, 8)
        data[:, :, 0] = 1
        data[:, :, 1] = 2
        data[:, :, 2] = 3
        data[:, :, 4] = 10
        data[:, :, 5] = 20
        data[:, :, 6] = 30
        
        config = {"p_phase_shuffling": 1.0}
        augmenter = TimeSeriesAugmenter(config)
        
        # Run multiple times to catch randomness (shift 1 or 2)
        # Since we can't control random seed easily inside, we check if it's EITHER shift 1 or 2
        output = augmenter(data)
        
        # Shift 1: A->B, B->C, C->A
        # IA=3, IB=1, IC=2
        shift1_currents = torch.tensor([3., 1., 2.])
        shift1_voltages = torch.tensor([30., 10., 20.])
        
        # Shift 2: A->C, B->A, C->B
        # IA=2, IB=3, IC=1
        shift2_currents = torch.tensor([2., 3., 1.])
        shift2_voltages = torch.tensor([20., 30., 10.])
        
        currents = output[0, 0, :3]
        voltages = output[0, 0, 4:7]
        
        is_shift1 = torch.allclose(currents, shift1_currents) and torch.allclose(voltages, shift1_voltages)
        is_shift2 = torch.allclose(currents, shift2_currents) and torch.allclose(voltages, shift2_voltages)
        
        assert is_shift1 or is_shift2

    def test_drop_channel(self, sample_data):
        config = {"p_drop_channel": 1.0}
        augmenter = TimeSeriesAugmenter(config)
        output = augmenter(sample_data)
        
        # Check that at least one channel is zero
        # Sum over time for each channel
        channel_sums = output.sum(dim=1) # (Batch, Channels)
        # Check if any channel is 0
        has_zero_channel = (channel_sums == 0).any()
        assert has_zero_channel

class TestDatasetAugmentation:
    
    @pytest.fixture
    def sample_df(self):
        # Create a dummy dataframe with 8 channels
        length = 100
        t = np.arange(length) / 1600.0
        f = 50.0
        
        # Balanced 3-phase system
        ia = np.sin(2 * np.pi * f * t)
        ib = np.sin(2 * np.pi * f * t - 2*np.pi/3)
        ic = np.sin(2 * np.pi * f * t + 2*np.pi/3)
        
        data = {
            'IA': ia,
            'IB': ib,
            'IC': ic,
            'IN': np.zeros(length), # Zero sequence is 0
            'UA': ia, # Just reuse currents for voltages
            'UB': ib,
            'UC': ic,
            'UN': np.zeros(length),
            'target': np.zeros(length)
        }
        return pl.DataFrame(data)

    def test_dataset_augmentation_integration(self, sample_df):
        indices = [0]
        window_size = 50
        
        # Config that inverts signal
        aug_config = {"p_inversion": 1.0}
        
        ds = OscillogramDataset(
            sample_df, 
            indices, 
            window_size, 
            feature_mode='raw',
            augmentation_config=aug_config,
            target_columns='target'
        )
        
        x, y = ds[0]
        # x shape: (Channels, Time)
        # Original data is sine. Inverted should be -sine.
        
        # Reconstruct original expected data (first 50 points)
        t = np.arange(window_size) / 1600.0
        f = 50.0
        expected_ia = -np.sin(2 * np.pi * f * t)
        
        # Check IA (channel 0)
        assert torch.allclose(x[0, :], torch.tensor(expected_ia, dtype=torch.float32), atol=1e-5)
        
    def test_dataset_augmentation_symmetric(self, sample_df):
        indices = [0]
        window_size = 50
        
        # Config that scales signal by 2
        aug_config = {
            "p_scaling": 1.0, 
            "scaling_range_current": (2.0, 2.0),
            "scaling_range_voltage": (2.0, 2.0)
        }
        
        ds = OscillogramDataset(
            sample_df, 
            indices, 
            window_size, 
            feature_mode='symmetric',
            augmentation_config=aug_config,
            target_columns='target'
        )
        
        x, y = ds[0]
        # Output format: [I1_re, I1_im, I2_re, I2_im, I0_re, I0_im, U1..., U2..., U0...]
        
        # Check Positive Sequence Current I1 (indices 0, 1)
        # Magnitude should be 2.0 (Amplitude)
        i1_re = x[0, -1]
        i1_im = x[1, -1]
        i1_mag = torch.sqrt(i1_re**2 + i1_im**2)
        
        # sliding_window_fft applies Hanning window which attenuates amplitude by 0.5
        # So for input amplitude 2.0, output should be ~1.0
        # Let's verify this by calculating expected value directly
        from osc_tools.features.pdr_calculator import sliding_window_fft
        
        t = np.arange(window_size) / 1600.0
        f = 50.0
        # Expected signal is scaled by 2.0
        expected_signal = 2.0 * np.sin(2 * np.pi * f * t)
        
        # Calculate FFT of expected signal
        # We need a longer signal to get valid output at the end
        # sliding_window_fft returns valid at index >= window_size
        # So let's create a signal of length window_size + 1
        signal_long = np.concatenate([expected_signal, [0]]) # Just one more point
        # Actually sliding_window_fft needs input > window_size to return anything useful?
        # No, if input == window_size, it returns 1 window.
        # But the result array has size of input.
        # Valid result is at index `window_size`.
        # Wait, sliding_window_fft returns array of same length as input.
        # Values [0 : window_size] are NaN.
        # Value at [window_size] corresponds to window [0 : window_size].
        
        # So we need input of length window_size + 1 to get value at index window_size?
        # Let's check implementation:
        # start_idx = window_size
        # fft_results[start_idx : start_idx + count] = harmonics[:count]
        
        # So if len(signal) == window_size, count = 0. No result.
        # We need len(signal) >= window_size + 1 to get at least 1 result at index window_size.
        
        # But ds[0] returns `x` of shape (Channels, window_size).
        # And `OscillogramDataset` calls `sliding_window_fft` on `raw_data` which has length `window_size`.
        # Wait, if `raw_data` has length `window_size`, `sliding_window_fft` returns ALL NaNs?
        
        # Let's check `OscillogramDataset` again.
        # `sample_df = self.data.slice(start_idx, self.window_size)`
        # `raw_data` has length `window_size`.
        # `sliding_window_fft(raw_data[:, i], fft_window, 1)`
        # `fft_window` is 32. `window_size` is 50.
        # So `raw_data` (50) > `fft_window` (32).
        # So it works.
        
        # So I should calculate expected value using `sliding_window_fft` with same parameters.
        fft_window = int(1600 / 50) # 32
        expected_phasor = sliding_window_fft(expected_signal, fft_window, 1)
        # The valid values start at index 32.
        # We want the value at index 49 (last point).
        expected_mag = np.abs(expected_phasor[-1, 0])
        
        print(f"DEBUG: Expected magnitude = {expected_mag}")
        
        assert torch.isclose(i1_mag, torch.tensor(expected_mag, dtype=torch.float32), atol=0.1)
        
        # Check Zero Sequence Current I0 (indices 4, 5)
        # Should be 0
        i0_re = x[4, -1]
        i0_im = x[5, -1]
        i0_mag = torch.sqrt(i0_re**2 + i0_im**2)
        assert torch.isclose(i0_mag, torch.tensor(0.0), atol=0.1)
