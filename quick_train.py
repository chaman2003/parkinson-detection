"""
Quick Train - Fast model training with minimal samples for testing
"""

import numpy as np
import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def quick_train():
    """Train simple models quickly for testing"""
    logger.info("Quick Training Mode - Creating lightweight models...")
    
    # Generate small synthetic datasets
    n_samples = 200
    n_voice_features = 150
    n_tremor_features = 200
    
    # Voice data
    X_voice = np.random.randn(n_samples, n_voice_features)
    y_voice = np.random.randint(0, 2, n_samples)
    
    # Tremor data
    X_tremor = np.random.randn(n_samples, n_tremor_features)
    y_tremor = np.random.randint(0, 2, n_samples)
    
    # Train simple models
    os.makedirs('backend/models', exist_ok=True)
    
    logger.info("Training voice model...")
    voice_scaler = StandardScaler()
    X_voice_scaled = voice_scaler.fit_transform(X_voice)
    voice_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    voice_model.fit(X_voice_scaled, y_voice)
    
    logger.info("Training tremor model...")
    tremor_scaler = StandardScaler()
    X_tremor_scaled = tremor_scaler.fit_transform(X_tremor)
    tremor_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    tremor_model.fit(X_tremor_scaled, y_tremor)
    
    # Save models
    logger.info("Saving models...")
    with open('backend/models/voice_model.pkl', 'wb') as f:
        pickle.dump(voice_model, f)
    with open('backend/models/voice_scaler.pkl', 'wb') as f:
        pickle.dump(voice_scaler, f)
    with open('backend/models/tremor_model.pkl', 'wb') as f:
        pickle.dump(tremor_model, f)
    with open('backend/models/tremor_scaler.pkl', 'wb') as f:
        pickle.dump(tremor_scaler, f)
    
    logger.info("✅ Models created and saved successfully!")
    logger.info("\nYou can now start the backend server:")
    logger.info("  python backend/app.py")


if __name__ == '__main__':
    try:
        quick_train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
