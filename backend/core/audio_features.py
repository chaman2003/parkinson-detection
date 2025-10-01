"""
Advanced Audio Feature Extraction for Parkinson's Disease Detection
Extracts voice and speech features known to be affected in Parkinson's patients
"""

import numpy as np
import librosa
import logging
from scipy import signal
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract comprehensive audio features for Parkinson's detection"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def extract_all_features(self, audio_path):
        """
        Extract all audio features from a given audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Dictionary of feature arrays and statistics
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Skip trimming to avoid numba JIT compilation issues
            y_trimmed = y
            
            if len(y_trimmed) == 0:
                logger.warning("Audio file is empty or all silence")
                return self._get_empty_features()
            
            features = {}
            
            # 1. MFCC Features (Mel-frequency cepstral coefficients)
            features.update(self._extract_mfcc_features(y_trimmed, sr))
            
            # 2. Spectral Features
            features.update(self._extract_spectral_features(y_trimmed, sr))
            
            # 3. Prosodic Features (pitch, energy, rhythm)
            features.update(self._extract_prosodic_features(y_trimmed, sr))
            
            # 4. Voice Quality Features
            features.update(self._extract_voice_quality_features(y_trimmed, sr))
            
            # 5. Temporal Features
            features.update(self._extract_temporal_features(y_trimmed, sr))
            
            # 6. Harmonic Features
            features.update(self._extract_harmonic_features(y_trimmed, sr))
            
            logger.info(f"Extracted {len(features)} audio features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            return self._get_empty_features()
    
    def _extract_mfcc_features(self, y, sr):
        """Extract MFCC features - critical for voice characterization"""
        features = {}
        
        # Standard MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Statistical moments of MFCCs
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
            features[f'mfcc_{i}_range'] = np.ptp(mfccs[i])
            features[f'mfcc_{i}_skew'] = skew(mfccs[i])
            features[f'mfcc_{i}_kurtosis'] = kurtosis(mfccs[i])
        
        # Delta MFCCs (velocity)
        mfcc_delta = librosa.feature.delta(mfccs)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        
        # Delta-Delta MFCCs (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2)
        features['mfcc_delta2_std'] = np.std(mfcc_delta2)
        
        return features
    
    def _extract_spectral_features(self, y, sr):
        """Extract spectral features - important for voice quality"""
        features = {}
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_max'] = np.max(spectral_centroids)
        features['spectral_centroid_min'] = np.min(spectral_centroids)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # Spectral flatness (measure of tonality vs. noisiness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # Zero crossing rate (indicates noisy vs. harmonic content)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def _extract_prosodic_features(self, y, sr):
        """Extract prosodic features - pitch, energy, rhythm"""
        features = {}
        
        # Pitch (F0) estimation using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_min'] = np.min(f0_clean)
            features['pitch_max'] = np.max(f0_clean)
            features['pitch_range'] = np.ptp(f0_clean)
            features['pitch_skew'] = skew(f0_clean)
            features['pitch_kurtosis'] = kurtosis(f0_clean)
            
            # Pitch variation (jitter)
            if len(f0_clean) > 1:
                pitch_diff = np.diff(f0_clean)
                features['pitch_jitter'] = np.mean(np.abs(pitch_diff))
                features['pitch_jitter_std'] = np.std(pitch_diff)
        else:
            # Default values if pitch cannot be extracted
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
                       'pitch_range', 'pitch_skew', 'pitch_kurtosis',
                       'pitch_jitter', 'pitch_jitter_std']:
                features[key] = 0.0
        
        # Voice probability
        features['voiced_prob_mean'] = np.mean(voiced_probs)
        features['voiced_prob_std'] = np.std(voiced_probs)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # Energy variation (shimmer)
        if len(rms) > 1:
            rms_diff = np.diff(rms)
            features['energy_shimmer'] = np.mean(np.abs(rms_diff))
            features['energy_shimmer_std'] = np.std(rms_diff)
        else:
            features['energy_shimmer'] = 0.0
            features['energy_shimmer_std'] = 0.0
        
        return features
    
    def _extract_voice_quality_features(self, y, sr):
        """Extract voice quality features - important for Parkinson's"""
        features = {}
        
        # Harmonic-to-Noise Ratio (HNR)
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Calculate HNR
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        
        if percussive_energy > 0:
            features['hnr'] = 10 * np.log10(harmonic_energy / percussive_energy)
        else:
            features['hnr'] = 20.0  # High HNR if no noise
        
        # Harmonic energy ratio
        total_energy = np.sum(y ** 2)
        if total_energy > 0:
            features['harmonic_ratio'] = harmonic_energy / total_energy
        else:
            features['harmonic_ratio'] = 0.0
        
        # Cepstral Peak Prominence (CPP) - measure of voice quality
        # Calculate cepstrum
        spectrum = np.abs(np.fft.rfft(y))
        cepstrum = np.abs(np.fft.rfft(np.log(spectrum + 1e-10)))
        
        if len(cepstrum) > 10:
            features['cpp'] = np.max(cepstrum[10:]) - np.mean(cepstrum[10:])
        else:
            features['cpp'] = 0.0
        
        return features
    
    def _extract_temporal_features(self, y, sr):
        """Extract temporal features - timing and rhythm"""
        features = {}
        
        try:
            # Onset detection (syllable/word timing)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features['onset_strength_mean'] = np.mean(onset_env)
            features['onset_strength_std'] = np.std(onset_env)
        except:
            features['onset_strength_mean'] = 0.0
            features['onset_strength_std'] = 0.0
        
        try:
            # Tempo estimation (can cause numba JIT issues, skip if problematic)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
        except:
            features['tempo'] = 120.0  # Default tempo
        
        # Autocorrelation (periodicity)
        autocorr = librosa.autocorrelate(y)
        features['autocorr_max'] = np.max(autocorr[1:100]) if len(autocorr) > 100 else 0.0
        
        # Duration features
        features['duration'] = len(y) / sr
        
        return features
    
    def _extract_harmonic_features(self, y, sr):
        """Extract harmonic features"""
        features = {}
        
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        return features
    
    def _get_empty_features(self):
        """Return default feature values when extraction fails"""
        return {
            'mfcc_0_mean': 0.0, 'spectral_centroid_mean': 0.0,
            'pitch_mean': 0.0, 'hnr': 0.0, 'duration': 0.0
        }
    
    def features_to_vector(self, features_dict):
        """Convert feature dictionary to numpy array"""
        # Sort keys for consistent ordering
        sorted_keys = sorted(features_dict.keys())
        return np.array([features_dict[key] for key in sorted_keys])
