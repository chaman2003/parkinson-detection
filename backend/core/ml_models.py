"""
Comprehensive ML Pipeline for Parkinson's Disease Detection
"""

import numpy as np
import os
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb

from .audio_features import AudioFeatureExtractor
from .tremor_features import TremorFeatureExtractor

logger = logging.getLogger(__name__)


class ParkinsonMLPipeline:
    """Complete ML Pipeline for Parkinson's Detection"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.audio_extractor = AudioFeatureExtractor()
        self.tremor_extractor = TremorFeatureExtractor()
        self.voice_model = None
        self.tremor_model = None
        self.voice_scaler = None
        self.tremor_scaler = None
        self.use_gpu = False  # Can be enabled for GPU training
        self.load_models()
    
    def build_ensemble_model(self, model_type='voice'):
        svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42)
        
        # Note: XGBoost removed temporarily due to sklearn 1.7.2 compatibility issue
        # The ensemble with SVM, RF, and GB still provides excellent performance
        ensemble = VotingClassifier(
            estimators=[('svm', svm_model), ('rf', rf_model), ('gb', gb_model)], 
            voting='soft', 
            weights=[1, 2, 2]
        )
        logger.info(f"Built {model_type} ensemble model (SVM + RF + GB)")
        return ensemble
    
    def train_models(self, X_voice, y_voice, X_tremor, y_tremor):
        logger.info("Training voice model...")
        self.voice_scaler = StandardScaler()
        X_voice_scaled = self.voice_scaler.fit_transform(X_voice)
        self.voice_model = self.build_ensemble_model('voice')
        self.voice_model.fit(X_voice_scaled, y_voice)
        voice_scores = cross_val_score(self.voice_model, X_voice_scaled, y_voice, cv=5)
        logger.info(f"Voice model CV accuracy: {voice_scores.mean():.3f}")
        
        logger.info("Training tremor model...")
        self.tremor_scaler = StandardScaler()
        X_tremor_scaled = self.tremor_scaler.fit_transform(X_tremor)
        self.tremor_model = self.build_ensemble_model('tremor')
        self.tremor_model.fit(X_tremor_scaled, y_tremor)
        tremor_scores = cross_val_score(self.tremor_model, X_tremor_scaled, y_tremor, cv=5)
        logger.info(f"Tremor model CV accuracy: {tremor_scores.mean():.3f}")
        
        self.save_models()
        logger.info("Training complete!")
    
    def analyze(self, audio_path, motion_data):
        import time
        start_time = time.time()
        
        audio_features = self.audio_extractor.extract_all_features(audio_path)
        tremor_features = self.tremor_extractor.extract_all_features(motion_data)
        
        audio_vector = self._dict_to_vector(audio_features)
        tremor_vector = self._dict_to_vector(tremor_features)
        
        voice_pred, voice_conf = self._predict_voice(audio_vector)
        tremor_pred, tremor_conf = self._predict_tremor(tremor_vector)
        
        combined_conf = (voice_conf * 0.5 + tremor_conf * 0.5)
        combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
        
        key_features = self._extract_key_features(audio_features, tremor_features)
        
        return {
            'prediction': combined_pred,
            'confidence': float(combined_conf),
            'voice_confidence': float(voice_conf),
            'tremor_confidence': float(tremor_conf),
            'features': key_features,
            'metadata': {
                'processing_time': round(time.time() - start_time, 2),
                'audio_features_count': len(audio_features),
                'tremor_features_count': len(tremor_features),
                'motion_samples': len(motion_data),
                'model_version': '1.0.0',
                'model_type': 'ensemble_ml'
            }
        }
    
    def _predict_voice(self, feature_vector):
        if self.voice_model is None or self.voice_scaler is None:
            return "Unknown", 0.5
        try:
            X = feature_vector.reshape(1, -1)
            X_scaled = self.voice_scaler.transform(X)
            prediction = self.voice_model.predict(X_scaled)[0]
            probabilities = self.voice_model.predict_proba(X_scaled)[0]
            confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            pred_label = "Affected" if prediction == 1 else "Not Affected"
            return pred_label, confidence
        except Exception as e:
            logger.error(f"Error in voice prediction: {str(e)}")
            return "Unknown", 0.5
    
    def _predict_tremor(self, feature_vector):
        if self.tremor_model is None or self.tremor_scaler is None:
            return "Unknown", 0.5
        try:
            X = feature_vector.reshape(1, -1)
            X_scaled = self.tremor_scaler.transform(X)
            prediction = self.tremor_model.predict(X_scaled)[0]
            probabilities = self.tremor_model.predict_proba(X_scaled)[0]
            confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            pred_label = "Affected" if prediction == 1 else "Not Affected"
            return pred_label, confidence
        except Exception as e:
            logger.error(f"Error in tremor prediction: {str(e)}")
            return "Unknown", 0.5
    
    def _dict_to_vector(self, features_dict):
        sorted_keys = sorted(features_dict.keys())
        return np.array([features_dict[key] for key in sorted_keys])
    
    def _extract_key_features(self, audio_features, tremor_features):
        key_features = {}
        if 'pitch_std' in audio_features:
            key_features['Voice Stability'] = round(1.0 - min(audio_features['pitch_std'] / 100.0, 1.0), 3)
        if 'hnr' in audio_features:
            key_features['Voice Quality'] = round(max(0, min(1, (audio_features['hnr'] + 10) / 30.0)), 3)
        if 'pitch_jitter' in audio_features:
            key_features['Vocal Tremor'] = round(1.0 - (1.0 - min(audio_features['pitch_jitter'] / 10.0, 1.0)), 3)
        if 'tremor_band_power_mag' in tremor_features:
            key_features['Tremor Frequency'] = round(min(tremor_features['tremor_band_power_mag'] / 100.0, 1.0), 3)
        if 'stability_index' in tremor_features:
            key_features['Postural Stability'] = round(1.0 - min(tremor_features['stability_index'], 1.0), 3)
        if 'magnitude_std' in tremor_features:
            key_features['Motion Variability'] = round(min(tremor_features['magnitude_std'] / 10.0, 1.0), 3)
        return key_features
    
    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.voice_model:
            with open(os.path.join(self.model_dir, 'voice_model.pkl'), 'wb') as f:
                pickle.dump(self.voice_model, f)
        if self.voice_scaler:
            with open(os.path.join(self.model_dir, 'voice_scaler.pkl'), 'wb') as f:
                pickle.dump(self.voice_scaler, f)
        if self.tremor_model:
            with open(os.path.join(self.model_dir, 'tremor_model.pkl'), 'wb') as f:
                pickle.dump(self.tremor_model, f)
        if self.tremor_scaler:
            with open(os.path.join(self.model_dir, 'tremor_scaler.pkl'), 'wb') as f:
                pickle.dump(self.tremor_scaler, f)
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        try:
            voice_model_path = os.path.join(self.model_dir, 'voice_model.pkl')
            if os.path.exists(voice_model_path):
                with open(voice_model_path, 'rb') as f:
                    self.voice_model = pickle.load(f)
                with open(os.path.join(self.model_dir, 'voice_scaler.pkl'), 'rb') as f:
                    self.voice_scaler = pickle.load(f)
                logger.info("Loaded voice model")
            
            tremor_model_path = os.path.join(self.model_dir, 'tremor_model.pkl')
            if os.path.exists(tremor_model_path):
                with open(tremor_model_path, 'rb') as f:
                    self.tremor_model = pickle.load(f)
                with open(os.path.join(self.model_dir, 'tremor_scaler.pkl'), 'rb') as f:
                    self.tremor_scaler = pickle.load(f)
                logger.info("Loaded tremor model")
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")
