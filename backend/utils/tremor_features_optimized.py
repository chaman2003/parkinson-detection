"""
Optimized Tremor Feature Extraction - Fast & Accurate
Focuses on the 25 core features with real-time insights
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import logging

logger = logging.getLogger(__name__)


class OptimizedTremorExtractor:
    """Fast tremor feature extraction with optimized algorithms"""
    
    def __init__(self):
        self.tremor_freq_range = (4, 6)  # Parkinson's tremor frequency band
        self.min_samples = 30  # Minimum samples required
    
    def _sanitize_features(self, features):
        """Replace NaN/Inf values with 0.0 to ensure valid JSON serialization"""
        sanitized = {}
        for key, value in features.items():
            if key.startswith('_'):
                # Keep metadata as-is
                sanitized[key] = value
            elif isinstance(value, (float, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    sanitized[key] = 0.0
                else:
                    sanitized[key] = float(value)
            elif isinstance(value, (int, np.integer)):
                sanitized[key] = int(value)
            else:
                sanitized[key] = value
        return sanitized
        
    def extract_features_fast(self, motion_data):
        """
        Extract 25 core tremor features efficiently
        Provides real-time insights based on actual input data
        """
        try:
            if len(motion_data) < self.min_samples:
                logger.warning(f"Insufficient samples: {len(motion_data)}")
                return self._get_default_features()
            
            # Parse motion data efficiently
            timestamps, accel_x, accel_y, accel_z = self._parse_fast(motion_data)
            
            if len(accel_x) < self.min_samples:
                return self._get_default_features()
            
            # Calculate magnitude vector (combines all axes)
            magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            
            # Check for idle/baseline state (phone sitting still on desk)
            is_idle, idle_metrics = self._detect_idle_baseline(magnitude, accel_x, accel_y, accel_z)
            
            if is_idle:
                logger.warning(f"⚠️ IDLE/BASELINE DETECTED: Std={idle_metrics['std']:.4f}, "
                             f"Range={idle_metrics['range']:.4f}, "
                             f"Variation={idle_metrics['variation_percent']:.2f}%")
                # Return minimal features indicating idle/baseline state
                features = self._get_default_features()
                features['_idle_detected'] = True
                features['_idle_metrics'] = idle_metrics
                return features
            
            # Extract sampling rate from timestamps
            if len(timestamps) > 1:
                dt_values = np.diff(timestamps) / 1000.0  # Convert ms to seconds
                sampling_rate = 1.0 / np.mean(dt_values) if np.mean(dt_values) > 0 else 50.0
            else:
                sampling_rate = 50.0  # Default
            
            features = {}
            
            # 1. Magnitude Statistics (12 features - most important)
            features.update(self._extract_magnitude_stats(magnitude))
            
            # 2. Frequency Domain Features (8 features - tremor detection)
            features.update(self._extract_frequency_features(magnitude, accel_x, accel_y, accel_z, sampling_rate))
            
            # 3. Time Domain Features (5 features - movement patterns)
            features.update(self._extract_time_features(magnitude, sampling_rate))
            
            # Mark as active motion detected
            features['_idle_detected'] = False
            
            # Sanitize all features to remove NaN/Inf values
            features = self._sanitize_features(features)
            
            logger.info(f"✓ Fast extraction: {len(features)} tremor features from {len(motion_data)} samples")
            
            # Real-time insights
            insights = self._generate_insights(features, sampling_rate)
            features['_insights'] = insights
            
            return features
            
        except Exception as e:
            logger.error(f"Fast tremor extraction failed: {e}")
            return self._get_default_features()
    
    def _detect_idle_baseline(self, magnitude, accel_x, accel_y, accel_z):
        """
        Detect if phone is idle/sitting still on desk with no meaningful motion
        Returns (is_idle, metrics_dict)
        """
        try:
            # Calculate variation metrics
            mag_std = np.std(magnitude)
            mag_range = np.max(magnitude) - np.min(magnitude)
            mag_mean = np.mean(magnitude)
            
            # Calculate coefficient of variation (CV)
            cv = (mag_std / mag_mean * 100) if mag_mean > 0 else 0
            
            # Check individual axes variation
            x_std = np.std(accel_x)
            y_std = np.std(accel_y)
            z_std = np.std(accel_z)
            total_std = x_std + y_std + z_std
            
            # Calculate approximate gravity (should be ~9.8 m/s² when still)
            # If magnitude is close to gravity with low variation, it's idle
            gravity_diff = abs(mag_mean - 9.8)
            
            # Detect idle if:
            # 1. Very low standard deviation (< 0.1 m/s²) AND low total axis variation
            # 2. Small range of motion (< 0.3 m/s²) AND low total axis variation
            # 3. Low coefficient of variation (< 2%)
            # 4. Or magnitude close to gravity with minimal variation
            
            # Key fix: Ensure we don't flag constant-magnitude motion (like perfect circles) as idle
            # by checking individual axis variation
            
            is_idle = (
                (mag_std < 0.15 and total_std < 0.5) or  # Stable magnitude AND stable axes
                (mag_range < 0.4 and total_std < 0.5) or  # Small range AND stable axes
                (cv < 3 and total_std < 0.5) or  # Low variation AND stable axes
                (gravity_diff < 0.5 and mag_std < 0.2 and total_std < 0.5)  # Still, near gravity
            )
            
            metrics = {
                'std': float(mag_std),
                'range': float(mag_range),
                'mean': float(mag_mean),
                'variation_percent': float(cv),
                'total_axis_std': float(total_std),
                'gravity_diff': float(gravity_diff),
                'x_std': float(x_std),
                'y_std': float(y_std),
                'z_std': float(z_std)
            }
            
            return is_idle, metrics
            
        except Exception as e:
            logger.warning(f"Idle detection error: {e}")
            # If detection fails, assume not idle to avoid false negatives
            return False, {}
    
    def _parse_fast(self, motion_data):
        """Parse motion data efficiently"""
        timestamps = []
        accel_x = []
        accel_y = []
        accel_z = []
        
        for sample in motion_data:
            if not isinstance(sample, dict):
                continue
            
            # Support both formats: {x, y, z} and {accelerationX, accelerationY, accelerationZ}
            x = sample.get('x', sample.get('accelerationX', 0))
            y = sample.get('y', sample.get('accelerationY', 0))
            z = sample.get('z', sample.get('accelerationZ', 0))
            t = sample.get('timestamp', 0)
            
            # Validate numeric values
            if all(isinstance(val, (int, float)) for val in [x, y, z, t]):
                timestamps.append(t)
                accel_x.append(x)
                accel_y.append(y)
                accel_z.append(z)
        
        return (np.array(timestamps), np.array(accel_x), 
                np.array(accel_y), np.array(accel_z))
    
    def _extract_magnitude_stats(self, magnitude):
        """Extract magnitude-based features (12 features)"""
        features = {}
        
        # Basic statistics - using lowercase for consistency
        features['magnitude_mean'] = float(np.mean(magnitude))
        features['magnitude_std'] = float(np.std(magnitude))
        features['magnitude_rms'] = float(np.sqrt(np.mean(magnitude**2)))
        features['magnitude_energy'] = float(np.sum(magnitude**2))
        
        # Peak detection
        peaks, _ = signal.find_peaks(magnitude, distance=5)
        features['magnitude_peaks_rt'] = float(len(peaks) / len(magnitude))
        
        # Slope sign changes (smoothness indicator)
        magnitude_diff = np.diff(magnitude)
        sign_changes = np.sum(np.diff(np.sign(magnitude_diff)) != 0)
        features['magnitude_ssc_rt'] = float(sign_changes / len(magnitude))
        
        # Additional magnitude metrics
        features['magnitude_max'] = float(np.max(magnitude))
        features['magnitude_min'] = float(np.min(magnitude))
        features['magnitude_range'] = features['magnitude_max'] - features['magnitude_min']
        features['magnitude_kurtosis'] = float(self._safe_kurtosis(magnitude))
        features['magnitude_skewness'] = float(self._safe_skew(magnitude))
        features['magnitude_cv'] = float(features['magnitude_std'] / (features['magnitude_mean'] + 1e-6))
        
        return features
    
    def _extract_frequency_features(self, magnitude, accel_x, accel_y, accel_z, sampling_rate):
        """Extract frequency domain features (8 features)"""
        features = {}
        
        try:
            # FFT on magnitude
            fft_mag = rfft(magnitude)
            freqs = rfftfreq(len(magnitude), d=1.0/sampling_rate)
            power_spectrum = np.abs(fft_mag)**2
            
            # Dominant frequency - lowercase for consistency
            if len(power_spectrum) > 1:
                dom_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
                features['magnitude_fft_dom_freq'] = float(freqs[dom_freq_idx])
            else:
                features['magnitude_fft_dom_freq'] = 0.0
            
            # Total power
            features['magnitude_fft_tot_power'] = float(np.sum(power_spectrum))
            
            # Energy in frequency domain
            features['magnitude_fft_energy'] = float(np.sum(power_spectrum))
            
            # Spectral entropy
            power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            features['magnitude_fft_entropy'] = float(entropy)
            
            # Tremor band power (4-6 Hz)
            tremor_mask = (freqs >= self.tremor_freq_range[0]) & (freqs <= self.tremor_freq_range[1])
            tremor_power = np.sum(power_spectrum[tremor_mask])
            total_power = np.sum(power_spectrum) + 1e-10
            features['tremor_band_power_mag'] = float(tremor_power / total_power)
            
            # Peak frequency in tremor band
            if np.any(tremor_mask):
                tremor_freqs = freqs[tremor_mask]
                tremor_powers = power_spectrum[tremor_mask]
                if len(tremor_powers) > 0:
                    features['tremor_peak_freq'] = float(tremor_freqs[np.argmax(tremor_powers)])
                else:
                    features['tremor_peak_freq'] = 5.0  # Middle of tremor band
            else:
                features['tremor_peak_freq'] = 5.0
            
            # Dominant frequency for X-axis (additional insight)
            fft_x = rfft(accel_x)
            power_x = np.abs(fft_x)**2
            if len(power_x) > 1:
                dom_freq_x_idx = np.argmax(power_x[1:]) + 1
                features['dominant_freq_x'] = float(freqs[dom_freq_x_idx])
            else:
                features['dominant_freq_x'] = 0.0
            
            # Tremor band power for X-axis
            tremor_power_x = np.sum(power_x[tremor_mask])
            features['tremor_band_power_x'] = float(tremor_power_x / (np.sum(power_x) + 1e-10))
            
        except Exception as e:
            logger.warning(f"Frequency extraction issue: {e}")
            features['magnitude_fft_dom_freq'] = 0.0
            features['magnitude_fft_tot_power'] = 0.0
            features['magnitude_fft_energy'] = 0.0
            features['magnitude_fft_entropy'] = 0.0
            features['tremor_band_power_mag'] = 0.0
            features['tremor_peak_freq'] = 5.0
            features['dominant_freq_x'] = 0.0
            features['tremor_band_power_x'] = 0.0
        
        return features
    
    def _extract_time_features(self, magnitude, sampling_rate):
        """Extract time domain features (6 features)"""
        features = {}
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0)
        features['zero_crossing_rate_mag'] = float(zero_crossings / len(magnitude))
        
        # Peak count
        peaks, _ = signal.find_peaks(magnitude, distance=int(sampling_rate * 0.1))
        features['peak_count_mag'] = float(len(peaks))
        
        # Jerk (rate of change of acceleration)
        jerk = np.diff(magnitude) * sampling_rate
        features['jerk_mean'] = float(np.mean(np.abs(jerk)))
        features['jerk_std'] = float(np.std(jerk))
        
        # Stability index (coefficient of variation)
        features['stability_index'] = float(np.std(magnitude) / (np.mean(magnitude) + 1e-6))
        
        # Sample entropy (complexity measure)
        features['magnitude_sampen'] = float(self._calculate_sample_entropy(magnitude))
        
        return features
    
    def _generate_insights(self, features, sampling_rate):
        """Generate real-time insights from extracted features"""
        insights = {
            'sampling_rate': round(sampling_rate, 1),
            'magnitude_level': 'normal',
            'tremor_detected': False,
            'tremor_strength': 'none',
            'movement_stability': 'stable',
            'frequency_analysis': 'normal'
        }
        
        # Magnitude analysis
        mag_mean = features.get('magnitude_mean', 0)
        if mag_mean > 15:
            insights['magnitude_level'] = 'high'
        elif mag_mean > 10:
            insights['magnitude_level'] = 'moderate'
        
        # Tremor detection
        tremor_power = features.get('tremor_band_power_mag', 0)
        tremor_freq = features.get('magnitude_fft_dom_freq', 0)
        
        if tremor_power > 0.15 and 4 <= tremor_freq <= 6:
            insights['tremor_detected'] = True
            if tremor_power > 0.25:
                insights['tremor_strength'] = 'strong'
            elif tremor_power > 0.15:
                insights['tremor_strength'] = 'moderate'
            else:
                insights['tremor_strength'] = 'mild'
        
        # Stability analysis
        stability_idx = features.get('stability_index', 0)
        if stability_idx > 0.5:
            insights['movement_stability'] = 'unstable'
        elif stability_idx > 0.3:
            insights['movement_stability'] = 'moderate'
        
        # Frequency characteristics
        if 4 <= tremor_freq <= 6:
            insights['frequency_analysis'] = 'tremor range (4-6 Hz)'
        elif tremor_freq > 0:
            insights['frequency_analysis'] = f'{tremor_freq:.1f} Hz'
        
        return insights
    
    def _safe_kurtosis(self, data):
        """Calculate kurtosis safely"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data, fisher=True, nan_policy='omit')
        except:
            return 0.0
    
    def _safe_skew(self, data):
        """Calculate skewness safely"""
        try:
            from scipy.stats import skew
            return skew(data, nan_policy='omit')
        except:
            return 0.0
    
    def _calculate_sample_entropy(self, data, m=2, r=None):
        """
        Calculate sample entropy (complexity measure)
        Higher values = more complex/irregular signal
        """
        try:
            N = len(data)
            if N < 10:
                return 0.0
            
            # Default tolerance
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) 
                     for i in range(len(x))]
                return sum(C)
            
            return -np.log(_phi(m + 1) / max(_phi(m), 1))
        except:
            return 1.0  # Default moderate entropy
    
    def _get_default_features(self):
        """Return default 25 features - all lowercase for consistency"""
        return {
            # Magnitude stats (12)
            'magnitude_mean': 0.0,
            'magnitude_std': 0.0,
            'magnitude_rms': 0.0,
            'magnitude_energy': 0.0,
            'magnitude_peaks_rt': 0.0,
            'magnitude_ssc_rt': 0.0,
            'magnitude_max': 0.0,
            'magnitude_min': 0.0,
            'magnitude_range': 0.0,
            'magnitude_kurtosis': 0.0,
            'magnitude_skewness': 0.0,
            'magnitude_cv': 0.0,
            
            # Frequency features (8)
            'magnitude_fft_dom_freq': 0.0,
            'magnitude_fft_tot_power': 0.0,
            'magnitude_fft_energy': 0.0,
            'magnitude_fft_entropy': 0.0,
            'tremor_band_power_mag': 0.0,
            'tremor_peak_freq': 5.0,
            'dominant_freq_x': 0.0,
            'tremor_band_power_x': 0.0,
            
            # Time features (6)
            'zero_crossing_rate_mag': 0.0,
            'peak_count_mag': 0.0,
            'jerk_mean': 0.0,
            'jerk_std': 0.0,
            'stability_index': 0.0,
            'magnitude_sampen': 0.0,
            
            '_insights': {
                'sampling_rate': 0,
                'magnitude_level': 'insufficient data',
                'tremor_detected': False,
                'tremor_strength': 'unknown',
                'movement_stability': 'unknown',
                'frequency_analysis': 'no data'
            }
        }
