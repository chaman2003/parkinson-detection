"""
Comprehensive ML Pipeline for Parkinson's Disease Detection
"""

import numpy as np
import os
import pickle
import logging
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Add parent directory to path for custom_scaler import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_scaler import CustomStandardScaler
from feature_mapper import map_features_to_training_format, features_dict_to_array

from .audio_features_optimized import OptimizedAudioExtractor
from .tremor_features_optimized import OptimizedTremorExtractor
from .personalized_model import PersonalizedModelHandler

logger = logging.getLogger(__name__)


class ParkinsonMLPipeline:
    """Complete ML Pipeline for Parkinson's Detection"""
    
    def __init__(self, model_dir='models', user_id=None):
        self.model_dir = model_dir
        self.user_id = user_id  # NEW: Support for personalized models
        self.audio_extractor = OptimizedAudioExtractor()
        self.tremor_extractor = OptimizedTremorExtractor()
        self.personalized_handler = PersonalizedModelHandler()  # NEW
        self.voice_model = None
        self.tremor_model = None
        self.voice_scaler = None
        self.tremor_scaler = None
        self.voice_feature_names = None
        self.use_gpu = False  # Can be enabled for GPU training
        self.load_models()
    
    def build_ensemble_model(self, model_type='voice'):
        svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42)
        
        estimators = [('svm', svm_model), ('rf', rf_model), ('gb', gb_model)]
        
        # Add XGBoost if available
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
            estimators.append(('xgb', xgb_model))
            logger.info(f"Added XGBoost to {model_type} ensemble")
        except Exception as e:
            logger.warning(f"XGBoost not available: {e}")
            
        ensemble = VotingClassifier(
            estimators=estimators, 
            voting='soft'
        )
        logger.info(f"Built {model_type} ensemble model")
        return ensemble
    
    def train_models(self, X_voice, y_voice, X_tremor, y_tremor):
        logger.info("Training voice model...")
        self.voice_scaler = CustomStandardScaler()
        X_voice_scaled = self.voice_scaler.fit_transform(X_voice)
        self.voice_model = self.build_ensemble_model('voice')
        self.voice_model.fit(X_voice_scaled, y_voice)
        voice_scores = cross_val_score(self.voice_model, X_voice_scaled, y_voice, cv=5)
        logger.info(f"Voice model CV accuracy: {voice_scores.mean():.3f}")
        
        logger.info("Training tremor model...")
        self.tremor_scaler = CustomStandardScaler()
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
        
        # Initialize vectors to None
        audio_vector = None
        tremor_vector = None
        voice_insights = {}
        tremor_insights = {}
        personalized_result = None  # NEW: Store personalized analysis
        
        # Flags for idle/baseline detection
        voice_is_idle = False
        tremor_is_idle = False
        
        # Extract features only if data is provided - using optimized extractors
        if audio_path is not None:
            audio_features = self.audio_extractor.extract_features_fast(audio_path)
            # Extract insights before converting to vector
            voice_insights = audio_features.pop('_insights', {})
            
            # Check for silence/idle detection
            voice_is_idle = audio_features.pop('_silence_detected', False)
            silence_metrics = audio_features.pop('_silence_metrics', {})
            
            # NEW: Try personalized model first if user_id is set
            if self.user_id and self.personalized_handler.has_personalized_model(self.user_id):
                logger.info(f"🎯 Using personalized model for user {self.user_id}")
                personalized_result = self.personalized_handler.analyze_with_personalized_model(
                    self.user_id, audio_features.copy()
                )
                
                if personalized_result.get('success'):
                    # Convert deviation to confidence score (inverse relationship)
                    # High deviation = low confidence (more different from baseline = potential issue)
                    deviation = personalized_result['deviation_percent']
                    personalized_confidence = min(100, max(0, 100 - deviation))
                    
                    logger.info(f"📊 Personalized: Deviation={deviation:.1f}%, Confidence={personalized_confidence:.1f}%")
                    
                    # Add personalized insights
                    voice_insights['personalized_analysis'] = {
                        'deviation_from_baseline': round(deviation, 1),
                        'is_anomaly': personalized_result['is_anomaly'],
                        'interpretation': personalized_result['interpretation'],
                        'baseline_samples': personalized_result['baseline_samples']
                    }
            
            # Create audio vector using ONLY selected features if available
            if self.voice_feature_names:
                # Keep a copy of ALL features for display purposes
                all_audio_features = audio_features.copy()
                
                audio_vector = self._dict_to_vector_selected(audio_features, self.voice_feature_names)
                # Filter features for response but keep the full set for raw_features
                filtered_audio_features = {k: audio_features.get(k, 0.0) for k in self.voice_feature_names}
            else:
                # Fallback to all features (should not happen if trained correctly)
                all_audio_features = audio_features.copy()
                filtered_audio_features = audio_features
                audio_vector = self._dict_to_vector(audio_features)
            
            # Check if vector is effectively empty/zeros (failed extraction)
            if np.all(np.abs(audio_vector) < 1e-6):
                logger.warning("⚠️ Audio vector is all zeros - treating as insufficient data")
                voice_is_idle = True
            
            if voice_is_idle:
                logger.warning(f"🔇 Voice idle/silence detected - returning low confidence")
                voice_conf = 0.05  # Very low confidence (5%)
                voice_pred = "Insufficient Data"
            else:
                voice_pred, voice_conf = self._predict_voice(audio_vector)
        else:
            audio_features = {}
            all_audio_features = {}
            filtered_audio_features = {}
            voice_conf = 0.5
            voice_pred = "Unknown"
            audio_vector = None
        
        if motion_data is not None:
            tremor_features = self.tremor_extractor.extract_features_fast(motion_data)
            # Extract insights before converting to vector
            tremor_insights = tremor_features.pop('_insights', {})
            
            # Check for idle/baseline detection
            tremor_is_idle = tremor_features.pop('_idle_detected', False)
            idle_metrics = tremor_features.pop('_idle_metrics', {})
            
            # ALWAYS map features to training format (12 features)
            mapped_tremor_features = map_features_to_training_format(tremor_features)
            
            # Convert to ordered array (12 features in correct order)
            tremor_vector = features_dict_to_array(mapped_tremor_features)
            
            if tremor_is_idle:
                logger.warning(f"📱 Motion idle/baseline detected - returning low confidence")
                tremor_conf = 0.05  # Very low confidence (5%)
                tremor_pred = "Insufficient Data"
            else:
                # Debug log tremor features
                logger.info(f"📊 Mapped to {len(mapped_tremor_features)} tremor features (training format)")
                logger.info(f"🔍 Key tremor values: Magnitude_mean={mapped_tremor_features.get('Magnitude_mean', 0):.2f}, "
                           f"Magnitude_std_dev={mapped_tremor_features.get('Magnitude_std_dev', 0):.2f}, "
                           f"Magnitude_fft_dom_freq={mapped_tremor_features.get('Magnitude_fft_dom_freq', 0):.2f}")
                logger.info(f"✅ Tremor vector shape: {tremor_vector.shape}")
                
                tremor_pred, tremor_conf = self._predict_tremor(tremor_vector)
        else:
            tremor_features = {}
            tremor_conf = 0.5
            tremor_pred = "Unknown"
            tremor_vector = None
        
        # Determine combined prediction based purely on ML models
        if audio_path is not None and motion_data is not None:
            # Both inputs provided
            if voice_is_idle and tremor_is_idle:
                # Both idle - very low confidence
                combined_conf = 0.03
                combined_pred = "Insufficient Data"
                logger.warning(f"⚠️ Both voice and motion are idle - returning {combined_conf*100:.1f}% confidence")
            elif voice_is_idle:
                # Voice idle, but tremor active - use tremor directly (no penalty)
                combined_conf = tremor_conf
                combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
                logger.warning(f"⚠️ Voice idle - using tremor only: {combined_conf*100:.1f}%")
            elif tremor_is_idle:
                # Tremor idle, but voice active - use voice directly (no penalty)
                # Voice analysis is reliable on its own for Parkinson's detection
                combined_conf = voice_conf
                combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
                logger.info(f"📊 Motion idle - using voice only: {combined_conf*100:.1f}%")
            else:
                # Both active - use raw ML predictions
                logger.info(f"📊 Raw ML confidences: voice={voice_conf:.3f}, tremor={tremor_conf:.3f}")
                
                # Combined confidence - simple average of both models
                combined_conf = (voice_conf + tremor_conf) / 2.0
                combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
                
                logger.info(f"✅ Final combined confidence: {combined_conf:.3f} ({combined_conf*100:.1f}%)")
            
        elif audio_path is not None:
            # Only voice available
            if voice_is_idle:
                combined_conf = 0.05
                combined_pred = "Insufficient Data"
                logger.warning(f"⚠️ Voice-only test with idle audio - returning {combined_conf*100:.1f}% confidence")
            else:
                # Use raw ML prediction
                combined_conf = voice_conf
                combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
                logger.info(f"📊 Voice-only raw ML confidence: {combined_conf:.3f} ({combined_conf*100:.1f}%)")
            
        elif motion_data is not None:
            # Only tremor available
            if tremor_is_idle:
                combined_conf = 0.05
                combined_pred = "Insufficient Data"
                logger.warning(f"⚠️ Tremor-only test with idle motion - returning {combined_conf*100:.1f}% confidence")
            else:
                # Use raw ML prediction
                combined_conf = tremor_conf
                combined_pred = "Affected" if combined_conf >= 0.5 else "Not Affected"
                logger.info(f"📊 Tremor-only raw ML confidence: {combined_conf:.3f} ({combined_conf*100:.1f}%)")
            
        else:
            # Neither available (shouldn't happen)
            combined_conf = 0.5
            combined_pred = "Not Affected"
        
        key_features = self._extract_key_features(all_audio_features, tremor_features)
        
        # Generate comprehensive insights
        comprehensive_insights = self._generate_comprehensive_insights(
            voice_insights, tremor_insights, voice_conf, tremor_conf, combined_conf
        )
        
        # NEW: Add personalized analysis to response if available
        if personalized_result and personalized_result.get('success'):
            comprehensive_insights['personalized'] = voice_insights.get('personalized_analysis', {})
        
        # Convert to 0-100 scale for user display
        # Calculate confidence of the PREDICTION (not just class 1 probability)
        # If prob is 0.05 (Healthy), confidence is 0.95 (95%)
        final_confidence = combined_conf if combined_conf >= 0.5 else (1.0 - combined_conf)
        
        # Calculate individual confidences similarly so they match the overall logic
        final_voice_confidence = voice_conf if voice_conf >= 0.5 else (1.0 - voice_conf)
        final_tremor_confidence = tremor_conf if tremor_conf >= 0.5 else (1.0 - tremor_conf)
        
        result = {
            'prediction': combined_pred,
            'confidence': round(float(final_confidence) * 100, 2),  # Confidence of the specific prediction
            'voice_confidence': round(float(voice_conf) * 100, 2),  # Probability of Parkinson's (0-100)
            'tremor_confidence': round(float(tremor_conf) * 100, 2),  # Probability of Parkinson's (0-100)
            'voice_patterns': round(float(final_voice_confidence) * 100, 2),  # Voice confidence (0-100)
            'motion_patterns': round(float(final_tremor_confidence) * 100, 2),  # Motion confidence (0-100)
            'features': key_features,  # Simplified 0-100 scale features for display
            'raw_features': {**all_audio_features, **tremor_features},  # All raw features for detailed view
            'tremor_features': tremor_features,  # Explicit tremor features
            'audio_features': all_audio_features,  # Explicit audio features (FULL SET)
            'insights': comprehensive_insights,  # Real-time insights
            'voice_features_vector': audio_vector,
            'tremor_features_vector': tremor_vector,
            'metadata': {
                'processing_time': round(time.time() - start_time, 2),
                'audio_features_count': len(all_audio_features),
                'tremor_features_count': len(tremor_features),
                'motion_samples': len(motion_data) if motion_data is not None else 0,
                'model_version': '1.0.0',
                'model_type': 'optimized_ensemble',
                'optimization': 'parallel_extraction',
                'uses_personalized_model': self.user_id is not None and personalized_result is not None
            }
        }
        
        # NEW: Add personalized metrics if available
        if personalized_result and personalized_result.get('success'):
            result['personalized_confidence'] = round(personalized_confidence, 2)
            result['baseline_deviation'] = round(personalized_result['deviation_percent'], 2)
        
        return result
    
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
    
    def _dict_to_vector_selected(self, features_dict, selected_names):
        """Convert feature dictionary to flat numpy array using ONLY selected features in order"""
        values = []
        for key in selected_names:
            val = features_dict.get(key, 0.0)
            # Convert to scalar if it's an array-like object
            if isinstance(val, (list, np.ndarray)):
                if hasattr(val, '__len__') and len(val) > 0:
                    val = float(np.mean(val))
                else:
                    val = 0.0
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            values.append(val)
        return np.array(values, dtype=np.float64)

    def _dict_to_vector(self, features_dict):
        """Convert feature dictionary to flat numpy array, handling nested arrays"""
        sorted_keys = sorted(features_dict.keys())
        values = []
        for key in sorted_keys:
            val = features_dict[key]
            # Convert to scalar if it's an array-like object
            if isinstance(val, (list, np.ndarray)):
                # If it's an array, take the mean or first value
                if hasattr(val, '__len__') and len(val) > 0:
                    val = float(np.mean(val))
                else:
                    val = 0.0
            # Ensure it's a scalar number
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            values.append(val)
        return np.array(values, dtype=np.float64)
    
    def _extract_key_features(self, audio_features, tremor_features):
        """Extract key features and return them as 0-100 scale percentages"""
        key_features = {}
        if 'pitch_std' in audio_features:
            key_features['Voice Stability'] = round((1.0 - min(audio_features['pitch_std'] / 100.0, 1.0)) * 100, 1)
        if 'hnr' in audio_features:
            key_features['Voice Quality'] = round(max(0, min(1, (audio_features['hnr'] + 10) / 30.0)) * 100, 1)
        if 'pitch_jitter' in audio_features:
            key_features['Vocal Tremor'] = round((1.0 - (1.0 - min(audio_features['pitch_jitter'] / 10.0, 1.0))) * 100, 1)
        # Map tremor features if needed
        if tremor_features:
            tremor_mapped = map_features_to_training_format(tremor_features) if 'magnitude_mean' in tremor_features else tremor_features
            
            if 'Magnitude_fft_entropy' in tremor_mapped:
                key_features['Tremor Frequency'] = round(min(tremor_mapped['Magnitude_fft_entropy'] / 10.0, 1.0) * 100, 1)
            if 'Magnitude_dfa' in tremor_mapped:
                key_features['Postural Stability'] = round((1.0 - min(tremor_mapped['Magnitude_dfa'], 1.0)) * 100, 1)
            if 'Magnitude_std_dev' in tremor_mapped:
                key_features['Motion Variability'] = round(min(tremor_mapped['Magnitude_std_dev'] / 10.0, 1.0) * 100, 1)
        return key_features
    
    def _generate_comprehensive_insights(self, voice_insights, tremor_insights, voice_conf, tremor_conf, combined_conf):
        """Generate comprehensive real-time insights from extracted features"""
        insights = {
            'summary': '',
            'voice_analysis': {},
            'tremor_analysis': {},
            'risk_level': 'low',
            'recommendations': []
        }
        
        # Determine risk level
        if combined_conf >= 0.75:
            insights['risk_level'] = 'high'
            insights['summary'] = 'High confidence detection - Multiple indicators present'
        elif combined_conf >= 0.55:
            insights['risk_level'] = 'moderate'
            insights['summary'] = 'Moderate confidence - Some indicators detected'
        else:
            insights['risk_level'] = 'low'
            insights['summary'] = 'Low confidence - Minimal indicators detected'
        
        # Voice analysis insights
        if voice_insights:
            insights['voice_analysis'] = {
                'audio_quality': voice_insights.get('audio_quality', 'unknown'),
                'voice_stability': voice_insights.get('voice_stability', 'unknown'),
                'pitch_characteristics': voice_insights.get('pitch_characteristics', 'unknown'),
                'harmonic_quality': voice_insights.get('harmonic_quality', 'unknown'),
                'confidence': round(voice_conf * 100, 1)
            }
            
            # Add voice-specific recommendations
            if voice_insights.get('jitter_level') == 'high':
                insights['recommendations'].append('Voice jitter detected - Consider vocal exercises')
            if voice_insights.get('audio_quality') == 'poor':
                insights['recommendations'].append('Audio quality low - Try recording in quieter environment')
        
        # Tremor analysis insights
        if tremor_insights:
            insights['tremor_analysis'] = {
                'tremor_detected': tremor_insights.get('tremor_detected', False),
                'tremor_strength': tremor_insights.get('tremor_strength', 'unknown'),
                'movement_stability': tremor_insights.get('movement_stability', 'unknown'),
                'frequency_analysis': tremor_insights.get('frequency_analysis', 'unknown'),
                'magnitude_level': tremor_insights.get('magnitude_level', 'unknown'),
                'sampling_rate': tremor_insights.get('sampling_rate', 0),
                'confidence': round(tremor_conf * 100, 1)
            }
            
            # Add tremor-specific recommendations
            if tremor_insights.get('tremor_detected'):
                insights['recommendations'].append(f"Tremor detected at {tremor_insights.get('tremor_strength', 'unknown')} strength")
            if tremor_insights.get('movement_stability') == 'unstable':
                insights['recommendations'].append('Movement instability detected - Consider stability exercises')
            if tremor_insights.get('sampling_rate', 0) < 20:
                insights['recommendations'].append('Low sampling rate - Hold device more steadily for better analysis')
        
        # General recommendations
        if not insights['recommendations']:
            insights['recommendations'].append('Continue regular monitoring for early detection')
        
        if combined_conf >= 0.6:
            insights['recommendations'].append('Consult healthcare professional for comprehensive evaluation')
        
        return insights
    
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
                
                # Load feature names if available
                feature_names_path = os.path.join(self.model_dir, 'voice_feature_names.pkl')
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'rb') as f:
                        self.voice_feature_names = pickle.load(f)
                    logger.info(f"Loaded {len(self.voice_feature_names)} selected voice features")
                
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
