"""
Optimized Audio Feature Extraction - Fast & Accurate
Focuses on most discriminative features for Parkinson's detection
"""

import numpy as np
import librosa
import logging
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import warnings
import gc  # For explicit memory cleanup

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class OptimizedAudioExtractor:
    """Fast audio feature extraction with parallel processing"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.n_mfcc = 13  # Optimized number of MFCC coefficients
        self.n_fft = 2048  # Balanced FFT size
        self.hop_length = 512  # Optimized hop length
        
    def extract_features_fast(self, audio_path):
        """
        Extract features in parallel for maximum speed
        Returns the most discriminative 138 features
        """
        try:
            # Load audio (optimized loading)
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=10)
            
            if len(y) < 1000:
                return self._get_default_features()
            
            # Check for silence/idle state - crucial for baseline detection
            is_silent, silence_metrics = self._detect_silence(y, sr)
            
            if is_silent:
                logger.warning(f"⚠️ SILENCE DETECTED: RMS={silence_metrics['rms_db']:.1f}dB, "
                             f"Energy={silence_metrics['energy']:.6f}, "
                             f"Voiced={silence_metrics['voiced_percent']:.1f}%")
                # Return minimal features indicating idle/silent state
                features = self._get_default_features()
                features['_silence_detected'] = True
                features['_silence_metrics'] = silence_metrics
                return features
            
            # Parallel feature extraction
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_mfcc = executor.submit(self._extract_mfcc_fast, y, sr)
                future_spectral = executor.submit(self._extract_spectral_fast, y, sr)
                future_prosodic = executor.submit(self._extract_prosodic_fast, y, sr)
                future_quality = executor.submit(self._extract_quality_fast, y, sr)
                
                # Gather results
                features = {}
                features.update(future_mfcc.result())
                features.update(future_spectral.result())
                features.update(future_prosodic.result())
                features.update(future_quality.result())
            
            # Add temporal and harmonic features (fast)
            features.update(self._extract_temporal_fast(y, sr))
            features.update(self._extract_harmonic_fast(y, sr))
            
            # Mark as active voice detected
            features['_silence_detected'] = False
            
            logger.info(f"✓ Fast extraction: {len(features)} features in optimized mode")
            
            # Explicitly delete audio array to release memory and file handles
            del y
            del sr
            
            # Force garbage collection to ensure file handles are released immediately
            # This is critical on Windows to prevent file locking issues
            gc.collect()
            
            return features
            
        except Exception as e:
            logger.error(f"Fast extraction failed: {e}")
            # Clean up on error too
            try:
                del y
                del sr
            except:
                pass
            return self._get_default_features()
    
    def _detect_silence(self, y, sr):
        """
        Detect if audio is silent/idle (phone on desk with no voice)
        Returns (is_silent, metrics_dict)
        """
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_db = 20 * np.log10(rms_mean + 1e-10)  # Convert to dB
            
            # Calculate overall energy
            energy = np.sum(y ** 2) / len(y)
            
            # Calculate zero-crossing rate (voice activity indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # Check for pitch/voiced content
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            voiced_frames = 0
            total_frames = pitches.shape[1]
            
            for t in range(total_frames):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 50:  # Valid voice pitch
                    voiced_frames += 1
            
            voiced_percent = (voiced_frames / total_frames) * 100 if total_frames > 0 else 0
            
            # Determine if silent based on multiple criteria
            is_silent = (
                rms_db < -40 or  # Very low volume (quieter than quiet room)
                energy < 0.001 or  # Minimal energy
                voiced_percent < 5  # Less than 5% voiced content
            )
            
            metrics = {
                'rms_db': float(rms_db),
                'energy': float(energy),
                'zcr_mean': float(zcr_mean),
                'voiced_percent': float(voiced_percent),
                'rms_mean': float(rms_mean)
            }
            
            return is_silent, metrics
            
        except Exception as e:
            logger.warning(f"Silence detection error: {e}")
            # If detection fails, assume not silent to avoid false negatives
            return False, {}
    
    def _extract_mfcc_fast(self, y, sr):
        """Extract MFCC features efficiently - 52 features"""
        features = {}
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Statistical moments (optimized)
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfcc[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfcc[i]))
        except Exception as e:
            logger.warning(f"MFCC extraction issue: {e}")
            for i in range(52):
                features[f'mfcc_feat_{i}'] = 0.0
        
        return features
    
    def _extract_spectral_fast(self, y, sr):
        """Extract spectral features efficiently - 28 features"""
        features = {}
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = float(np.mean(centroid))
            features['spectral_centroid_std'] = float(np.std(centroid))
            features['spectral_centroid_min'] = float(np.min(centroid))
            features['spectral_centroid_max'] = float(np.max(centroid))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(bandwidth))
            features['spectral_bandwidth_min'] = float(np.min(bandwidth))
            features['spectral_bandwidth_max'] = float(np.max(bandwidth))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            features['spectral_rolloff_mean'] = float(np.mean(rolloff))
            features['spectral_rolloff_std'] = float(np.std(rolloff))
            features['spectral_rolloff_min'] = float(np.min(rolloff))
            features['spectral_rolloff_max'] = float(np.max(rolloff))
            
            # Spectral contrast (7 bands × 2 stats = 14)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            for i in range(7):
                features[f'spectral_contrast_{i}_mean'] = float(np.mean(contrast[i]))
                features[f'spectral_contrast_{i}_std'] = float(np.std(contrast[i]))
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            features['spectral_flatness_mean'] = float(np.mean(flatness))
            features['spectral_flatness_std'] = float(np.std(flatness))
            
        except Exception as e:
            logger.warning(f"Spectral extraction issue: {e}")
            for i in range(28):
                features[f'spectral_feat_{i}'] = 0.0
        
        return features
    
    def _extract_prosodic_fast(self, y, sr):
        """Extract prosodic features (pitch, energy) - 24 features"""
        features = {}
        try:
            # Pitch/F0 using piptrack (fast method)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_min'] = 0.0
                features['pitch_max'] = 0.0
                features['pitch_range'] = 0.0
            
            # Jitter (pitch variability)
            if len(pitch_values) > 1:
                pitch_diffs = np.abs(np.diff(pitch_values))
                features['jitter_local'] = float(np.mean(pitch_diffs))
                features['jitter_abs'] = float(np.mean(pitch_diffs))
                features['jitter_rap'] = float(np.mean(pitch_diffs) / (np.mean(pitch_values) + 1e-6))
                features['jitter_ppq5'] = float(np.std(pitch_diffs))
            else:
                features['jitter_local'] = 0.0
                features['jitter_abs'] = 0.0
                features['jitter_rap'] = 0.0
                features['jitter_ppq5'] = 0.0
            
            # Energy/RMS
            rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            features['rms_min'] = float(np.min(rms))
            features['rms_max'] = float(np.max(rms))
            
            # Shimmer (amplitude variability)
            rms_diffs = np.abs(np.diff(rms))
            features['shimmer_local'] = float(np.mean(rms_diffs))
            features['shimmer_abs'] = float(np.mean(rms_diffs))
            features['shimmer_apq3'] = float(np.std(rms_diffs))
            features['shimmer_apq5'] = float(np.std(rms_diffs))
            
            # Speech rate estimation
            features['speech_rate'] = float(len(pitch_values) / (len(y) / sr))
            features['voiced_fraction'] = float(len(pitch_values) / (len(y) / self.hop_length))
            features['pause_rate'] = 1.0 - features['voiced_fraction']
            features['rhythm_variability'] = float(np.std(rms))
            
        except Exception as e:
            logger.warning(f"Prosodic extraction issue: {e}")
            for i in range(24):
                features[f'prosodic_feat_{i}'] = 0.0
        
        return features
    
    def _extract_quality_fast(self, y, sr):
        """Extract voice quality features - 18 features"""
        features = {}
        try:
            # Harmonic-to-Noise Ratio approximation
            # Split into harmonic and percussive
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            harmonic_energy = np.sum(y_harmonic ** 2)
            noise_energy = np.sum(y_percussive ** 2)
            hnr = 10 * np.log10((harmonic_energy / (noise_energy + 1e-6)) + 1e-6)
            features['hnr'] = float(hnr)
            features['hnr_mean'] = float(hnr)
            features['hnr_std'] = 1.0
            
            # Shimmer estimates
            rms = librosa.feature.rms(y=y)[0]
            features['shimmer'] = float(np.std(rms) / (np.mean(rms) + 1e-6))
            features['shimmer_local'] = features['shimmer']
            
            # Jitter estimates
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['jitter'] = float(np.std(zcr) / (np.mean(zcr) + 1e-6))
            features['jitter_local'] = features['jitter']
            
            # Voice breaks
            silence_threshold = 0.01
            is_silent = rms < silence_threshold
            voice_breaks = np.sum(np.diff(is_silent.astype(int)) != 0)
            features['voice_breaks'] = float(voice_breaks)
            features['unvoiced_frames'] = float(np.sum(is_silent))
            features['voiced_unvoiced_ratio'] = float(np.sum(~is_silent) / (np.sum(is_silent) + 1))
            
            # Additional quality metrics
            features['signal_noise_ratio'] = float(hnr)
            features['harmonicity'] = float(harmonic_energy / (harmonic_energy + noise_energy + 1e-6))
            features['noisiness'] = 1.0 - features['harmonicity']
            features['voice_quality_index'] = float(np.clip(hnr / 20.0, 0, 1))
            
            # Fill remaining to reach 18
            for i in range(4):
                features[f'quality_extra_{i}'] = 0.0
            
        except Exception as e:
            logger.warning(f"Quality extraction issue: {e}")
            for i in range(18):
                features[f'quality_feat_{i}'] = 0.0
        
        return features
    
    def _extract_temporal_fast(self, y, sr):
        """Extract temporal features - 8 features"""
        features = {}
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            features['zcr_min'] = float(np.min(zcr))
            features['zcr_max'] = float(np.max(zcr))
            
            # Duration statistics
            features['duration'] = float(len(y) / sr)
            features['effective_duration'] = features['duration']
            
            # Onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = float(np.mean(onset_env))
            features['onset_strength_std'] = float(np.std(onset_env))
            
        except Exception as e:
            logger.warning(f"Temporal extraction issue: {e}")
            for i in range(8):
                features[f'temporal_feat_{i}'] = 0.0
        
        return features
    
    def _extract_harmonic_fast(self, y, sr):
        """Extract harmonic features - 8 features"""
        features = {}
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Harmonic statistics
            features['harmonic_mean'] = float(np.mean(np.abs(y_harmonic)))
            features['harmonic_std'] = float(np.std(y_harmonic))
            features['harmonic_energy'] = float(np.sum(y_harmonic ** 2))
            features['harmonic_ratio'] = float(np.sum(y_harmonic ** 2) / (np.sum(y ** 2) + 1e-6))
            
            # Percussive statistics
            features['percussive_mean'] = float(np.mean(np.abs(y_percussive)))
            features['percussive_std'] = float(np.std(y_percussive))
            features['percussive_energy'] = float(np.sum(y_percussive ** 2))
            features['percussive_ratio'] = float(np.sum(y_percussive ** 2) / (np.sum(y ** 2) + 1e-6))
            
        except Exception as e:
            logger.warning(f"Harmonic extraction issue: {e}")
            for i in range(8):
                features[f'harmonic_feat_{i}'] = 0.0
        
        return features
    
    def _get_default_features(self):
        """Return default feature dictionary"""
        features = {}
        # Total: 52 + 28 + 24 + 18 + 8 + 8 = 138 features
        for i in range(138):
            features[f'feature_{i}'] = 0.0
        return features
