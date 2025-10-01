"""
Advanced Tremor/Motion Feature Extraction for Parkinson's Disease Detection
Extracts movement features from accelerometer data known to be affected in Parkinson's patients
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)


class TremorFeatureExtractor:
    """Extract comprehensive tremor and motion features for Parkinson's detection"""
    
    def __init__(self, sampling_rate=50):
        """
        Initialize tremor feature extractor
        
        Args:
            sampling_rate: Expected sampling rate of motion data (Hz)
        """
        self.sampling_rate = sampling_rate
        
    def extract_all_features(self, motion_data):
        """
        Extract all tremor features from motion data
        
        Args:
            motion_data: List of dicts with keys: timestamp, accelerationX, accelerationY, accelerationZ
            
        Returns:
            dict: Dictionary of feature values
        """
        try:
            if len(motion_data) < 50:
                logger.warning(f"Insufficient motion data: {len(motion_data)} samples")
                return self._get_empty_features()
            
            # Parse motion data
            timestamps, accel_x, accel_y, accel_z = self._parse_motion_data(motion_data)
            
            if len(timestamps) == 0:
                return self._get_empty_features()
            
            features = {}
            
            # 1. Time Domain Features
            features.update(self._extract_time_domain_features(accel_x, accel_y, accel_z))
            
            # 2. Frequency Domain Features (critical for tremor detection)
            features.update(self._extract_frequency_domain_features(accel_x, accel_y, accel_z, timestamps))
            
            # 3. Statistical Features
            features.update(self._extract_statistical_features(accel_x, accel_y, accel_z))
            
            # 4. Tremor-Specific Features
            features.update(self._extract_tremor_features(accel_x, accel_y, accel_z, timestamps))
            
            # 5. Movement Pattern Features
            features.update(self._extract_movement_patterns(accel_x, accel_y, accel_z, timestamps))
            
            # 6. Stability Features
            features.update(self._extract_stability_features(accel_x, accel_y, accel_z))
            
            logger.info(f"Extracted {len(features)} tremor features from {len(motion_data)} samples")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting tremor features: {str(e)}")
            return self._get_empty_features()
    
    def _parse_motion_data(self, motion_data):
        """Parse motion data into separate arrays"""
        timestamps = []
        accel_x = []
        accel_y = []
        accel_z = []
        
        for sample in motion_data:
            if isinstance(sample, dict):
                try:
                    timestamps.append(float(sample.get('timestamp', 0)))
                    accel_x.append(float(sample.get('accelerationX', 0)))
                    accel_y.append(float(sample.get('accelerationY', 0)))
                    accel_z.append(float(sample.get('accelerationZ', 0)))
                except (ValueError, TypeError):
                    continue
        
        return (np.array(timestamps), np.array(accel_x), 
                np.array(accel_y), np.array(accel_z))
    
    def _extract_time_domain_features(self, accel_x, accel_y, accel_z):
        """Extract time domain features"""
        features = {}
        
        # Magnitude of acceleration vector
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Basic statistics for each axis
        for axis_name, axis_data in [('x', accel_x), ('y', accel_y), ('z', accel_z)]:
            features[f'accel_{axis_name}_mean'] = np.mean(axis_data)
            features[f'accel_{axis_name}_std'] = np.std(axis_data)
            features[f'accel_{axis_name}_max'] = np.max(axis_data)
            features[f'accel_{axis_name}_min'] = np.min(axis_data)
            features[f'accel_{axis_name}_range'] = np.ptp(axis_data)
            features[f'accel_{axis_name}_rms'] = np.sqrt(np.mean(axis_data**2))
        
        # Magnitude statistics
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_max'] = np.max(magnitude)
        features['magnitude_min'] = np.min(magnitude)
        features['magnitude_range'] = np.ptp(magnitude)
        
        # Signal energy
        features['energy_x'] = np.sum(accel_x**2)
        features['energy_y'] = np.sum(accel_y**2)
        features['energy_z'] = np.sum(accel_z**2)
        features['total_energy'] = features['energy_x'] + features['energy_y'] + features['energy_z']
        
        return features
    
    def _extract_frequency_domain_features(self, accel_x, accel_y, accel_z, timestamps):
        """Extract frequency domain features - critical for tremor detection"""
        features = {}
        
        # Estimate sampling rate from timestamps
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            avg_sample_interval = np.mean(time_diffs) / 1000.0  # Convert to seconds
            estimated_sr = 1.0 / avg_sample_interval if avg_sample_interval > 0 else self.sampling_rate
        else:
            estimated_sr = self.sampling_rate
        
        # Magnitude signal
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # FFT analysis for each axis and magnitude
        for signal_name, signal_data in [('x', accel_x), ('y', accel_y), ('z', accel_z), ('mag', magnitude)]:
            # Compute FFT
            n = len(signal_data)
            fft_vals = fft(signal_data)
            fft_freq = fftfreq(n, 1.0/estimated_sr)
            
            # Only positive frequencies
            pos_mask = fft_freq > 0
            freqs = fft_freq[pos_mask]
            power = np.abs(fft_vals[pos_mask])**2
            
            if len(freqs) > 0:
                # Dominant frequency (peak power)
                dominant_freq_idx = np.argmax(power)
                features[f'dominant_freq_{signal_name}'] = freqs[dominant_freq_idx]
                features[f'dominant_power_{signal_name}'] = power[dominant_freq_idx]
                
                # Parkinson's tremor typically 4-6 Hz range
                tremor_band = (freqs >= 4) & (freqs <= 6)
                if np.any(tremor_band):
                    features[f'tremor_band_power_{signal_name}'] = np.sum(power[tremor_band])
                    features[f'tremor_band_ratio_{signal_name}'] = np.sum(power[tremor_band]) / np.sum(power)
                else:
                    features[f'tremor_band_power_{signal_name}'] = 0.0
                    features[f'tremor_band_ratio_{signal_name}'] = 0.0
                
                # Rest tremor band (3-7 Hz)
                rest_tremor_band = (freqs >= 3) & (freqs <= 7)
                if np.any(rest_tremor_band):
                    features[f'rest_tremor_power_{signal_name}'] = np.sum(power[rest_tremor_band])
                else:
                    features[f'rest_tremor_power_{signal_name}'] = 0.0
                
                # Spectral centroid
                if np.sum(power) > 0:
                    features[f'spectral_centroid_{signal_name}'] = np.sum(freqs * power) / np.sum(power)
                else:
                    features[f'spectral_centroid_{signal_name}'] = 0.0
                
                # Spectral entropy
                power_norm = power / np.sum(power) if np.sum(power) > 0 else power
                power_norm = power_norm[power_norm > 0]
                if len(power_norm) > 0:
                    features[f'spectral_entropy_{signal_name}'] = -np.sum(power_norm * np.log2(power_norm))
                else:
                    features[f'spectral_entropy_{signal_name}'] = 0.0
        
        return features
    
    def _extract_statistical_features(self, accel_x, accel_y, accel_z):
        """Extract advanced statistical features"""
        features = {}
        
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        for signal_name, signal_data in [('x', accel_x), ('y', accel_y), ('z', accel_z), ('mag', magnitude)]:
            # Skewness (asymmetry of distribution)
            features[f'skewness_{signal_name}'] = stats.skew(signal_data)
            
            # Kurtosis (tailedness of distribution)
            features[f'kurtosis_{signal_name}'] = stats.kurtosis(signal_data)
            
            # Variance
            features[f'variance_{signal_name}'] = np.var(signal_data)
            
            # Coefficient of variation
            mean_val = np.mean(signal_data)
            if mean_val != 0:
                features[f'cv_{signal_name}'] = np.std(signal_data) / abs(mean_val)
            else:
                features[f'cv_{signal_name}'] = 0.0
            
            # Percentiles
            features[f'percentile_25_{signal_name}'] = np.percentile(signal_data, 25)
            features[f'percentile_75_{signal_name}'] = np.percentile(signal_data, 75)
            features[f'iqr_{signal_name}'] = stats.iqr(signal_data)
        
        return features
    
    def _extract_tremor_features(self, accel_x, accel_y, accel_z, timestamps):
        """Extract tremor-specific features"""
        features = {}
        
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Tremor amplitude (peak-to-peak)
        features['tremor_amplitude_x'] = np.ptp(accel_x)
        features['tremor_amplitude_y'] = np.ptp(accel_y)
        features['tremor_amplitude_z'] = np.ptp(accel_z)
        features['tremor_amplitude_magnitude'] = np.ptp(magnitude)
        
        # Zero crossing rate (indicates frequency of oscillation)
        for signal_name, signal_data in [('x', accel_x), ('y', accel_y), ('z', accel_z), ('mag', magnitude)]:
            zero_crossings = np.where(np.diff(np.sign(signal_data - np.mean(signal_data))))[0]
            features[f'zero_crossing_rate_{signal_name}'] = len(zero_crossings) / len(signal_data)
        
        # Jerk (rate of change of acceleration)
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps) / 1000.0  # Convert to seconds
            
            for axis_name, axis_data in [('x', accel_x), ('y', accel_y), ('z', accel_z)]:
                jerk = np.diff(axis_data) / time_diffs
                features[f'jerk_mean_{axis_name}'] = np.mean(np.abs(jerk))
                features[f'jerk_std_{axis_name}'] = np.std(jerk)
                features[f'jerk_max_{axis_name}'] = np.max(np.abs(jerk))
        
        # Signal smoothness (inverse of jerk)
        for axis_name, axis_data in [('x', accel_x), ('y', accel_y), ('z', accel_z)]:
            if len(axis_data) > 1:
                smoothness = -np.sum(np.diff(axis_data)**2)
                features[f'smoothness_{axis_name}'] = smoothness
        
        return features
    
    def _extract_movement_patterns(self, accel_x, accel_y, accel_z, timestamps):
        """Extract movement pattern features"""
        features = {}
        
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Autocorrelation (measure of periodicity/repetitiveness)
        for signal_name, signal_data in [('x', accel_x), ('y', accel_y), ('z', accel_z), ('mag', magnitude)]:
            if len(signal_data) > 10:
                autocorr = np.correlate(signal_data - np.mean(signal_data), 
                                       signal_data - np.mean(signal_data), 
                                       mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
                
                # First minimum after initial peak indicates tremor period
                if len(autocorr) > 5:
                    features[f'autocorr_first_min_{signal_name}'] = np.min(autocorr[1:min(20, len(autocorr))])
                    features[f'autocorr_decay_{signal_name}'] = autocorr[min(10, len(autocorr)-1)]
        
        # Movement intensity over time
        window_size = max(10, len(magnitude) // 10)
        if len(magnitude) >= window_size:
            windowed_intensity = []
            for i in range(0, len(magnitude) - window_size, window_size):
                window = magnitude[i:i+window_size]
                windowed_intensity.append(np.std(window))
            
            if len(windowed_intensity) > 0:
                features['intensity_variation'] = np.std(windowed_intensity)
                features['intensity_trend'] = np.polyfit(range(len(windowed_intensity)), windowed_intensity, 1)[0]
        
        return features
    
    def _extract_stability_features(self, accel_x, accel_y, accel_z):
        """Extract postural stability features"""
        features = {}
        
        magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Stability index (lower is more stable)
        features['stability_index'] = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
        
        # Root mean square of each axis (measures deviation from rest)
        features['rms_x'] = np.sqrt(np.mean(accel_x**2))
        features['rms_y'] = np.sqrt(np.mean(accel_y**2))
        features['rms_z'] = np.sqrt(np.mean(accel_z**2))
        
        # Total path length (cumulative movement)
        path_length = np.sum(np.abs(np.diff(magnitude)))
        features['path_length'] = path_length
        
        # Mean velocity (approximation)
        if len(magnitude) > 1:
            features['mean_velocity'] = path_length / len(magnitude)
        else:
            features['mean_velocity'] = 0.0
        
        # Axis dominance (which axis shows most movement)
        axis_stds = [np.std(accel_x), np.std(accel_y), np.std(accel_z)]
        total_std = sum(axis_stds)
        if total_std > 0:
            features['axis_dominance_x'] = axis_stds[0] / total_std
            features['axis_dominance_y'] = axis_stds[1] / total_std
            features['axis_dominance_z'] = axis_stds[2] / total_std
        else:
            features['axis_dominance_x'] = 1/3
            features['axis_dominance_y'] = 1/3
            features['axis_dominance_z'] = 1/3
        
        return features
    
    def _get_empty_features(self):
        """Return default feature values when extraction fails"""
        return {
            'magnitude_mean': 0.0,
            'dominant_freq_mag': 0.0,
            'tremor_band_power_mag': 0.0,
            'stability_index': 0.0
        }
    
    def features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy array"""
        sorted_keys = sorted(features_dict.keys())
        return np.array([features_dict[key] for key in sorted_keys])
