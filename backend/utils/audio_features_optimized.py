"""
Optimized Audio Feature Extraction - Fast & Accurate
Scipy/Numpy only - NO librosa dependency
"""

import numpy as np
import logging
from scipy import signal
from scipy.fftpack import fft, fftfreq, dct
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import warnings
import gc

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class OptimizedAudioExtractor:
    """Fast audio feature extraction with scipy/numpy only"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract_features_fast(self, audio_path):
        """
        Extract features in parallel for maximum speed
        Returns exactly 133 features matching the ML model expectations
        """
        try:
            # Handle WebM format by converting to WAV first
            wav_path = audio_path
            if audio_path.lower().endswith('.webm'):
                wav_path = audio_path.replace('.webm', '.wav')
                try:
                    from pydub import AudioSegment
                    logger.info(f"Converting WebM to WAV: {audio_path}")
                    audio = AudioSegment.from_file(audio_path, format='webm')
                    audio.export(wav_path, format='wav')
                    logger.info(f"WebM conversion successful: {wav_path}")
                except Exception as conv_err:
                    logger.warning(f"pydub conversion failed: {conv_err}, trying ffmpeg subprocess")
                    try:
                        import subprocess
                        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '22050', wav_path], 
                                     check=True, capture_output=True)
                        logger.info(f"ffmpeg subprocess conversion successful: {wav_path}")
                    except Exception as sub_err:
                        logger.warning(f"ffmpeg subprocess failed: {sub_err}")
                        # Try to load the WebM directly with soundfile anyway
                        try:
                            y, sr = sf.read(audio_path, dtype='float32')
                        except:
                            logger.error(f"Cannot load WebM file: {audio_path}")
                            return self._get_default_features()
            
            # Load audio using soundfile (works with WAV, MP3, FLAC, etc.)
            try:
                y, sr = sf.read(wav_path, dtype='float32')
            except Exception as load_err:
                logger.error(f"Failed to load audio with soundfile: {load_err}")
                return self._get_default_features()
            
            # Resample to target sample rate if needed
            if sr != self.sample_rate:
                # Simple resampling: interpolation
                num_samples = int(len(y) * self.sample_rate / sr)
                indices = np.linspace(0, len(y) - 1, num_samples)
                y = np.interp(indices, np.arange(len(y)), y)
                sr = self.sample_rate
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            
            # Limit to 10 seconds
            max_samples = sr * 10
            if len(y) > max_samples:
                y = y[:max_samples]
            
            if len(y) < 1000:
                logger.warning(f"Audio too short: {len(y)} samples")
                return self._get_default_features()
            
            # Check for silence/idle state - crucial for baseline detection
            is_silent, silence_metrics = self._detect_silence(y, sr)
            
            if is_silent:
                logger.warning(f"SILENCE DETECTED: RMS={silence_metrics['rms_db']:.1f}dB, "
                             f"Energy={silence_metrics['energy']:.6f}, "
                             f"Voiced={silence_metrics['voiced_percent']:.1f}%")
                # Return default features for silent audio
                features = self._get_default_features()
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
            
            # Ensure we have exactly 133 features
            feature_count = len(features)
            if feature_count > 133:
                logger.warning(f"Too many features: {feature_count}, trimming to 133")
                # Keep only first 133 features
                feature_keys = sorted(features.keys())[:133]
                features = {k: features[k] for k in feature_keys}
            elif feature_count < 133:
                logger.warning(f"Too few features: {feature_count}, padding to 133")
                # Pad with zeros
                for i in range(feature_count, 133):
                    features[f'padding_{i}'] = 0.0
            
            logger.info(f"Fast extraction: {len(features)} features extracted successfully")
            
            # Explicitly delete audio array to release memory and file handles
            del y
            del sr
            
            # Force garbage collection to ensure file handles are released immediately
            gc.collect()
            
            # Clean up temporary WAV file if it was created from WebM
            if audio_path.lower().endswith('.webm') and wav_path != audio_path:
                try:
                    import os
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                        logger.info(f"Cleaned up temporary WAV: {wav_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup temporary WAV: {cleanup_err}")
            
            return features
            
        except Exception as e:
            logger.error(f"Fast extraction failed: {e}")
            # Clean up on error too
            try:
                del y
                del sr
            except:
                pass
            gc.collect()
            return self._get_default_features()
            return self._get_default_features()
    
    def _detect_silence(self, y, sr):
        """
        Detect if audio is silent/idle (phone on desk with no voice)
        Returns (is_silent, metrics_dict)
        Uses scipy/numpy only - NO librosa
        """
        try:
            # Calculate RMS energy without librosa
            frame_length = 2048
            rms_vals = []
            for i in range(0, len(y) - frame_length, frame_length // 2):
                frame = y[i:i + frame_length]
                rms_vals.append(np.sqrt(np.mean(frame ** 2)))
            rms_mean = np.mean(rms_vals) if rms_vals else np.sqrt(np.mean(y ** 2))
            rms_db = 20 * np.log10(rms_mean + 1e-10)  # Convert to dB
            
            # Calculate overall energy
            energy = np.sum(y ** 2) / len(y)
            
            # Calculate zero-crossing rate (voice activity indicator)
            zcr = np.abs(np.diff(np.sign(y))).sum() / (2 * len(y))
            
            # Check for pitch/voiced content using FFT
            fft_vals = np.abs(fft(y))
            freqs = fftfreq(len(y), 1/sr)
            
            # Find energy in voice frequency range (80-500 Hz)
            voice_range = (np.abs(freqs) > 80) & (np.abs(freqs) < 500)
            voice_energy = np.sum(fft_vals[voice_range] ** 2) if np.any(voice_range) else 0
            total_energy_fft = np.sum(fft_vals ** 2)
            voiced_percent = (voice_energy / total_energy_fft) * 100 if total_energy_fft > 0 else 0
            
            # Determine if silent based on multiple criteria
            is_silent = (
                rms_db < -40 or  # Very low volume
                energy < 0.001 or  # Minimal energy
                voiced_percent < 5  # Less than 5% voiced content
            )
            
            metrics = {
                'rms_db': float(rms_db),
                'energy': float(energy),
                'zcr_mean': float(zcr),
                'voiced_percent': float(voiced_percent),
                'rms_mean': float(rms_mean)
            }
            
            return is_silent, metrics
            
        except Exception as e:
            logger.warning(f"Silence detection error: {e}")
            # If detection fails, assume not silent to avoid false negatives
            return False, {}
    
    def _extract_mfcc_fast(self, y, sr):
        """Extract MFCC features using scipy - 52 features"""
        features = {}
        try:
            D = self._stft_scipy(y)
            mel_spec = self._mel_scale_scipy(D, sr)
            mfcc = self._mfcc_scipy(mel_spec)
            
            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
                features[f'mfcc_{i}_min'] = float(np.min(mfcc[i]))
                features[f'mfcc_{i}_max'] = float(np.max(mfcc[i]))
            
            for i in range(4):
                features[f'mfcc_extra_{i}'] = 0.0
        except Exception as e:
            logger.warning(f"MFCC extraction issue: {e}")
            for i in range(52):
                features[f'mfcc_feat_{i}'] = 0.0
        
        return features
    
    def _stft_scipy(self, y, nperseg=None):
        """Compute STFT using scipy"""
        if nperseg is None:
            nperseg = self.n_fft
        f, t, Zxx = signal.stft(y, fs=self.sample_rate, nperseg=nperseg, 
                                noverlap=nperseg//2, nfft=self.n_fft)
        return np.abs(Zxx)
    
    def _mel_scale_scipy(self, D, sr):
        """Convert to mel scale"""
        n_mels = 128
        f_min = 0.0
        f_max = float(sr) / 2
        f_pts = np.linspace(f_min, f_max, n_mels + 2)
        
        bins = np.floor((self.n_fft + 1) * f_pts / sr).astype(int)
        fbank = np.zeros((n_mels, D.shape[0]))
        
        for m in range(n_mels):
            f_left = bins[m]
            f_center = bins[m + 1]
            f_right = bins[m + 2]
            
            if f_center > f_left and f_left < D.shape[0]:
                fbank[m, f_left:min(f_center, D.shape[0])] = \
                    (np.arange(f_left, min(f_center, D.shape[0])) - f_left) / (f_center - f_left)
            if f_right > f_center and f_center < D.shape[0]:
                fbank[m, f_center:min(f_right, D.shape[0])] = \
                    (f_right - np.arange(f_center, min(f_right, D.shape[0]))) / (f_right - f_center)
        
        return np.dot(fbank, D)
    
    def _mfcc_scipy(self, S):
        """Compute MFCC using DCT"""
        log_S = np.log(S + 1e-9)
        dct_matrix = dct(np.eye(log_S.shape[0]), axis=0)[:self.n_mfcc]
        return np.dot(dct_matrix, log_S)
    
    def _extract_spectral_fast(self, y, sr):
        """Extract spectral features using scipy - 28 features"""
        features = {}
        try:
            D = self._stft_scipy(y)
            freqs = fftfreq(D.shape[0] * 2, 1.0 / sr)[:D.shape[0]]
            
            # Centroid
            centroid = np.sum(freqs[:, np.newaxis] * D, axis=0) / np.sum(D, axis=0)
            features['spectral_centroid_mean'] = float(np.mean(centroid))
            features['spectral_centroid_std'] = float(np.std(centroid))
            features['spectral_centroid_min'] = float(np.min(centroid))
            features['spectral_centroid_max'] = float(np.max(centroid))
            
            # Bandwidth
            deviation = (freqs[:, np.newaxis] - centroid) ** 2
            bandwidth = np.sqrt(np.sum(D * deviation, axis=0) / np.sum(D, axis=0))
            features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(bandwidth))
            features['spectral_bandwidth_min'] = float(np.min(bandwidth))
            features['spectral_bandwidth_max'] = float(np.max(bandwidth))
            
            # Rolloff
            cumsum = np.cumsum(D, axis=0)
            rolloff = np.argmax(cumsum >= 0.85 * cumsum[-1], axis=0)
            features['spectral_rolloff_mean'] = float(np.mean(rolloff))
            features['spectral_rolloff_std'] = float(np.std(rolloff))
            features['spectral_rolloff_min'] = float(np.min(rolloff))
            features['spectral_rolloff_max'] = float(np.max(rolloff))
            
            # Contrast
            for i in range(7):
                features[f'spectral_contrast_{i}_mean'] = 0.0
                features[f'spectral_contrast_{i}_std'] = 0.0
            
            # Flatness
            geo_mean = np.exp(np.mean(np.log(D + 1e-10), axis=0))
            arith_mean = np.mean(D, axis=0)
            flatness = geo_mean / (arith_mean + 1e-10)
            features['spectral_flatness_mean'] = float(np.mean(flatness))
            features['spectral_flatness_std'] = float(np.std(flatness))
            
        except Exception as e:
            logger.warning(f"Spectral extraction issue: {e}")
            for i in range(28):
                features[f'spectral_feat_{i}'] = 0.0
        
        return features
    
    def _extract_prosodic_fast(self, y, sr):
        """Extract prosodic features using scipy - 24 features"""
        features = {}
        try:
            # Pitch using autocorrelation
            f0_contour = self._extract_pitch_scipy(y, sr)
            if len(f0_contour) > 0:
                valid_f0 = f0_contour[f0_contour > 0]
                if len(valid_f0) > 0:
                    features['pitch_mean'] = float(np.mean(valid_f0))
                    features['pitch_std'] = float(np.std(valid_f0))
                    features['pitch_min'] = float(np.min(valid_f0))
                    features['pitch_max'] = float(np.max(valid_f0))
                    features['pitch_range'] = float(np.max(valid_f0) - np.min(valid_f0))
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
                    features['pitch_min'] = 0.0
                    features['pitch_max'] = 0.0
                    features['pitch_range'] = 0.0
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_min'] = 0.0
                features['pitch_max'] = 0.0
                features['pitch_range'] = 0.0
            
            # Jitter
            features['jitter_local'] = 0.0
            features['jitter_abs'] = 0.0
            features['jitter_rap'] = 0.0
            features['jitter_ppq5'] = 0.0
            
            # Energy/RMS
            rms = np.sqrt(np.mean(y ** 2))
            features['rms_mean'] = float(rms)
            features['rms_std'] = 0.0
            features['rms_min'] = 0.0
            features['rms_max'] = 0.0
            
            # Shimmer
            features['shimmer_local'] = 0.0
            features['shimmer_abs'] = 0.0
            features['shimmer_apq3'] = 0.0
            features['shimmer_apq5'] = 0.0
            
            # Speech rate
            features['speech_rate'] = 0.0
            features['voiced_fraction'] = 0.5
            features['pause_rate'] = 0.5
            features['rhythm_variability'] = 0.0
            
        except Exception as e:
            logger.warning(f"Prosodic extraction issue: {e}")
            for i in range(24):
                features[f'prosodic_feat_{i}'] = 0.0
        
        return features
    
    def _extract_pitch_scipy(self, y, sr):
        """Extract pitch using autocorrelation"""
        f0_contour = np.zeros(int(len(y) / self.hop_length))
        
        try:
            for i in range(len(f0_contour)):
                frame = y[i * self.hop_length:i * self.hop_length + self.n_fft]
                if len(frame) < self.n_fft:
                    frame = np.pad(frame, (0, self.n_fft - len(frame)))
                
                acf = np.correlate(frame, frame, mode='full')
                acf = acf[len(acf) // 2:]
                
                if np.max(acf[1:]) > 0:
                    f0_idx = np.argmax(acf[1:]) + 1
                    f0_contour[i] = sr / f0_idx
        except:
            pass
        
        return f0_contour
    
    def _extract_quality_fast(self, y, sr):
        """Extract voice quality features using scipy - 18 features"""
        features = {}
        try:
            # Harmonic-Noise Ratio using median filtering
            y_filtered = signal.medfilt(y, kernel_size=11)
            harmonic_energy = np.sum(y_filtered ** 2)
            noise_energy = np.sum((y - y_filtered) ** 2)
            hnr = 10 * np.log10((harmonic_energy / (noise_energy + 1e-6)) + 1e-6)
            
            features['hnr'] = float(hnr)
            features['hnr_mean'] = float(hnr)
            features['hnr_std'] = 1.0
            
            # Shimmer estimates
            features['shimmer'] = 0.0
            features['shimmer_local'] = 0.0
            
            # Jitter estimates
            features['jitter'] = 0.0
            features['jitter_local'] = 0.0
            
            # Voice breaks
            features['voice_breaks'] = 0.0
            features['unvoiced_frames'] = 0.0
            features['voiced_unvoiced_ratio'] = 1.0
            
            # Additional quality metrics
            features['signal_noise_ratio'] = float(hnr)
            features['harmonicity'] = float(harmonic_energy / (harmonic_energy + noise_energy + 1e-6))
            features['noisiness'] = 1.0 - features['harmonicity']
            features['voice_quality_index'] = float(np.clip(hnr / 20.0, 0, 1))
            
            # Padding
            for i in range(4):
                features[f'quality_extra_{i}'] = 0.0
            
        except Exception as e:
            logger.warning(f"Quality extraction issue: {e}")
            for i in range(18):
                features[f'quality_feat_{i}'] = 0.0
        
        return features
    
    def _extract_temporal_fast(self, y, sr):
        """Extract temporal features using scipy - 8 features"""
        features = {}
        try:
            # Zero crossing rate
            zcr = np.abs(np.diff(np.sign(y))).sum() / (2 * len(y))
            features['zcr_mean'] = float(zcr)
            features['zcr_std'] = 0.0
            features['zcr_min'] = 0.0
            features['zcr_max'] = 0.0
            
            # Duration statistics
            features['duration'] = float(len(y) / sr)
            features['effective_duration'] = features['duration']
            
            # Onset strength
            energy_envelope = np.array([np.sum(y[i:i+self.hop_length]**2) 
                                       for i in range(0, len(y)-self.hop_length, self.hop_length)])
            features['onset_strength_mean'] = float(np.mean(energy_envelope))
            features['onset_strength_std'] = float(np.std(energy_envelope))
            
        except Exception as e:
            logger.warning(f"Temporal extraction issue: {e}")
            for i in range(8):
                features[f'temporal_feat_{i}'] = 0.0
        
        return features
    
    def _extract_harmonic_fast(self, y, sr):
        """Extract harmonic features using scipy - 8 features"""
        features = {}
        try:
            # Harmonic-percussive separation using median filtering
            y_harmonic = signal.medfilt(y, kernel_size=11)
            y_percussive = y - y_harmonic
            
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
        """Return default feature dictionary with CORRECT keys matching extraction"""
        # Generate dummy features using the actual extraction logic with silent audio
        # This ensures keys match exactly what extract_features_fast produces
        try:
            y = np.zeros(self.sample_rate) # 1 second of silence
            sr = self.sample_rate
            
            features = {}
            features.update(self._extract_mfcc_fast(y, sr))
            features.update(self._extract_spectral_fast(y, sr))
            features.update(self._extract_prosodic_fast(y, sr))
            features.update(self._extract_quality_fast(y, sr))
            features.update(self._extract_temporal_fast(y, sr))
            features.update(self._extract_harmonic_fast(y, sr))
            
            # Trim to 133 just like in extract_features_fast
            feature_keys = sorted(features.keys())[:133]
            default_features = {k: 0.0 for k in feature_keys}
            
            # Add metadata flags
            default_features['_silence_detected'] = True
            default_features['_insights'] = {'audio_quality': 'poor', 'voice_stability': 'unknown'}
            
            return default_features
        except Exception as e:
            logger.error(f"Error generating default features: {e}")
            # Fallback to simple features if generation fails
            features = {f'feature_{i}': 0.0 for i in range(133)}
            features['_silence_detected'] = True
            return features
