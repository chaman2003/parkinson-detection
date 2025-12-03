"""
Personalized Model Handler - User-Specific Baseline Calibration
================================================================
Allows users to record healthy baseline samples and create personalized models
"""

import os
import pickle
import numpy as np
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)


class PersonalizedModelHandler:
    """Manages personalized baseline models for individual users"""
    
    def __init__(self, storage_dir='user_profiles'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_baseline_sample(self, user_id, audio_features, sample_index):
        """
        Save a single baseline (healthy) audio sample for a user
        
        Args:
            user_id: Unique identifier for the user
            audio_features: Dictionary of extracted audio features
            sample_index: Index of this sample (0, 1, 2, etc.)
        """
        user_dir = os.path.join(self.storage_dir, user_id, 'baseline_samples')
        os.makedirs(user_dir, exist_ok=True)
        
        # Save features
        sample_path = os.path.join(user_dir, f'sample_{sample_index}.pkl')
        with open(sample_path, 'wb') as f:
            pickle.dump(audio_features, f)
        
        logger.info(f"Saved baseline sample {sample_index} for user {user_id}")
        return True
    
    def save_baseline_tremor_sample(self, user_id, tremor_features, sample_index):
        """
        Save a single baseline (healthy) tremor sample for a user
        
        Args:
            user_id: Unique identifier for the user
            tremor_features: Dictionary of extracted tremor features
            sample_index: Index of this sample (0, 1, 2, etc.)
        """
        user_dir = os.path.join(self.storage_dir, user_id, 'baseline_tremor_samples')
        os.makedirs(user_dir, exist_ok=True)
        
        # Save features
        sample_path = os.path.join(user_dir, f'tremor_sample_{sample_index}.pkl')
        with open(sample_path, 'wb') as f:
            pickle.dump(tremor_features, f)
        
        logger.info(f"Saved baseline tremor sample {sample_index} for user {user_id}")
        return True
    
    def get_baseline_samples(self, user_id):
        """Load all baseline samples for a user"""
        user_dir = os.path.join(self.storage_dir, user_id, 'baseline_samples')
        
        if not os.path.exists(user_dir):
            return []
        
        samples = []
        for filename in sorted(os.listdir(user_dir)):
            if filename.endswith('.pkl'):
                with open(os.path.join(user_dir, filename), 'rb') as f:
                    samples.append(pickle.load(f))
        
        return samples
    
    def get_baseline_tremor_samples(self, user_id):
        """Load all baseline tremor samples for a user"""
        user_dir = os.path.join(self.storage_dir, user_id, 'baseline_tremor_samples')
        
        if not os.path.exists(user_dir):
            return []
        
        samples = []
        for filename in sorted(os.listdir(user_dir)):
            if filename.endswith('.pkl'):
                with open(os.path.join(user_dir, filename), 'rb') as f:
                    samples.append(pickle.load(f))
        
        return samples
    
    def train_personalized_model(self, user_id, min_samples=3):
        """
        Train a personalized anomaly detection model from baseline samples
        
        Uses Isolation Forest to learn the user's healthy voice characteristics
        and detect deviations from their baseline
        
        Args:
            user_id: Unique identifier for the user
            min_samples: Minimum number of baseline samples required
            
        Returns:
            dict with status and message
        """
        baseline_samples = self.get_baseline_samples(user_id)
        
        if len(baseline_samples) < min_samples:
            return {
                'success': False,
                'message': f'Need at least {min_samples} baseline samples, got {len(baseline_samples)}'
            }
        
        try:
            # Extract feature vectors from samples
            # Remove metadata keys
            feature_vectors = []
            for sample in baseline_samples:
                clean_sample = {k: v for k, v in sample.items() 
                              if not k.startswith('_')}
                # Sort keys for consistency
                sorted_keys = sorted(clean_sample.keys())
                vector = np.array([clean_sample[k] for k in sorted_keys], dtype=np.float64)
                feature_vectors.append(vector)
            
            X_baseline = np.array(feature_vectors)
            
            # Train scaler on baseline data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_baseline)
            
            # Train Isolation Forest (anomaly detector)
            # contamination=0.1 means we expect ~10% of new samples might be anomalies
            model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                max_samples='auto'
            )
            model.fit(X_scaled)
            
            # Calculate baseline statistics for additional context
            baseline_stats = {
                'mean': np.mean(X_baseline, axis=0).tolist(),
                'std': np.std(X_baseline, axis=0).tolist(),
                'min': np.min(X_baseline, axis=0).tolist(),
                'max': np.max(X_baseline, axis=0).tolist(),
                'feature_names': sorted_keys,
                'num_samples': len(baseline_samples),
                'trained_at': datetime.now().isoformat()
            }
            
            # Save model and metadata
            user_dir = os.path.join(self.storage_dir, user_id)
            model_path = os.path.join(user_dir, 'personalized_model.pkl')
            scaler_path = os.path.join(user_dir, 'personalized_scaler.pkl')
            stats_path = os.path.join(user_dir, 'baseline_stats.json')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(stats_path, 'w') as f:
                json.dump(baseline_stats, f, indent=2)
            
            # Also train tremor model if tremor samples exist
            tremor_samples = self.get_baseline_tremor_samples(user_id)
            tremor_trained = False
            if len(tremor_samples) >= min_samples:
                tremor_trained = self._train_tremor_model(user_id, tremor_samples)
            
            logger.info(f"Trained personalized model for user {user_id} with {len(baseline_samples)} voice samples")
            
            return {
                'success': True,
                'message': f'Successfully trained personalized model with {len(baseline_samples)} voice samples' + 
                          (f' and {len(tremor_samples)} tremor samples' if tremor_trained else ''),
                'stats': baseline_stats,
                'tremor_trained': tremor_trained
            }
            
        except Exception as e:
            logger.error(f"Error training personalized model for {user_id}: {e}")
            return {
                'success': False,
                'message': f'Training failed: {str(e)}'
            }
    
    def _train_tremor_model(self, user_id, tremor_samples):
        """Train a personalized tremor model"""
        try:
            feature_vectors = []
            sorted_keys = None
            
            for sample in tremor_samples:
                clean_sample = {k: v for k, v in sample.items() 
                              if not k.startswith('_') and isinstance(v, (int, float))}
                if sorted_keys is None:
                    sorted_keys = sorted(clean_sample.keys())
                vector = np.array([clean_sample.get(k, 0) for k in sorted_keys], dtype=np.float64)
                feature_vectors.append(vector)
            
            X_baseline = np.array(feature_vectors)
            
            # Train scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_baseline)
            
            # Train Isolation Forest
            model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                max_samples='auto'
            )
            model.fit(X_scaled)
            
            # Save tremor model
            user_dir = os.path.join(self.storage_dir, user_id)
            with open(os.path.join(user_dir, 'tremor_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            with open(os.path.join(user_dir, 'tremor_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save tremor stats
            tremor_stats = {
                'mean': np.mean(X_baseline, axis=0).tolist(),
                'std': np.std(X_baseline, axis=0).tolist(),
                'feature_names': sorted_keys,
                'num_samples': len(tremor_samples),
                'trained_at': datetime.now().isoformat()
            }
            with open(os.path.join(user_dir, 'tremor_stats.json'), 'w') as f:
                json.dump(tremor_stats, f, indent=2)
            
            logger.info(f"Trained personalized tremor model for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error training tremor model: {e}")
            return False
    
    def has_personalized_model(self, user_id):
        """Check if a user has a trained personalized model"""
        model_path = os.path.join(self.storage_dir, user_id, 'personalized_model.pkl')
        return os.path.exists(model_path)
    
    def load_personalized_model(self, user_id):
        """Load a user's personalized model and scaler"""
        if not self.has_personalized_model(user_id):
            return None, None, None
        
        user_dir = os.path.join(self.storage_dir, user_id)
        model_path = os.path.join(user_dir, 'personalized_model.pkl')
        scaler_path = os.path.join(user_dir, 'personalized_scaler.pkl')
        stats_path = os.path.join(user_dir, 'baseline_stats.json')
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            return model, scaler, stats
        except Exception as e:
            logger.error(f"Error loading personalized model for {user_id}: {e}")
            return None, None, None
    
    def analyze_with_personalized_model(self, user_id, audio_features):
        """
        Analyze new audio sample using personalized model
        
        Returns:
            dict with anomaly_score, is_anomaly, deviation_percent
        """
        model, scaler, stats = self.load_personalized_model(user_id)
        
        if model is None:
            return {
                'success': False,
                'message': 'No personalized model found'
            }
        
        try:
            # Prepare features (same processing as baseline)
            clean_features = {k: v for k, v in audio_features.items() 
                            if not k.startswith('_')}
            sorted_keys = sorted(clean_features.keys())
            feature_vector = np.array([clean_features[k] for k in sorted_keys], dtype=np.float64)
            
            # Scale features
            X_scaled = scaler.transform(feature_vector.reshape(1, -1))
            
            # Get anomaly score (-1 = anomaly, 1 = normal)
            prediction = model.predict(X_scaled)[0]
            # Get decision function score (lower = more anomalous)
            anomaly_score = model.decision_function(X_scaled)[0]
            
            # Calculate deviation from baseline
            baseline_mean = np.array(stats['mean'])
            baseline_std = np.array(stats['std'])
            
            # Z-score: how many standard deviations away from baseline
            z_scores = (feature_vector - baseline_mean) / (baseline_std + 1e-6)
            avg_z_score = np.mean(np.abs(z_scores))
            
            # Convert to percentage deviation (0-100%)
            # Higher deviation = more different from baseline
            deviation_percent = min(100, avg_z_score * 20)  # Scale factor to make it intuitive
            
            is_anomaly = prediction == -1
            
            return {
                'success': True,
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'deviation_percent': float(deviation_percent),
                'baseline_samples': stats['num_samples'],
                'trained_at': stats['trained_at'],
                'interpretation': self._interpret_deviation(deviation_percent, is_anomaly)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing with personalized model: {e}")
            return {
                'success': False,
                'message': f'Analysis failed: {str(e)}'
            }
    
    def _interpret_deviation(self, deviation_percent, is_anomaly):
        """Provide human-readable interpretation of deviation"""
        if deviation_percent < 10:
            return "Voice matches your baseline - Very Consistent"
        elif deviation_percent < 25:
            return "Minor variation from baseline - Normal"
        elif deviation_percent < 50:
            return "Moderate variation detected - Monitor"
        else:
            return "Significant variation from baseline - Consult Healthcare Professional"
    
    def get_user_status(self, user_id):
        """Get calibration status for a user"""
        baseline_samples = self.get_baseline_samples(user_id)
        tremor_samples = self.get_baseline_tremor_samples(user_id)
        has_model = self.has_personalized_model(user_id)
        
        status = {
            'user_id': user_id,
            'has_baseline_samples': len(baseline_samples) > 0,
            'baseline_sample_count': len(baseline_samples),
            'tremor_sample_count': len(tremor_samples),
            'has_trained_model': has_model,
            'is_calibrated': has_model
        }
        
        if has_model:
            stats_path = os.path.join(self.storage_dir, user_id, 'baseline_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    status['trained_at'] = stats.get('trained_at')
                    status['num_baseline_samples'] = stats.get('num_samples')
            
            # Check for tremor model
            tremor_stats_path = os.path.join(self.storage_dir, user_id, 'tremor_stats.json')
            if os.path.exists(tremor_stats_path):
                with open(tremor_stats_path, 'r') as f:
                    tremor_stats = json.load(f)
                    status['tremor_trained_at'] = tremor_stats.get('trained_at')
                    status['num_tremor_samples'] = tremor_stats.get('num_samples')
        
        return status
    
    def delete_user_data(self, user_id):
        """Delete all data for a user (for privacy/reset)"""
        import shutil
        user_dir = os.path.join(self.storage_dir, user_id)
        
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
            logger.info(f"Deleted all data for user {user_id}")
            return {'success': True, 'message': 'User data deleted'}
        
        return {'success': False, 'message': 'User not found'}
