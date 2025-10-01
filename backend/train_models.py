"""
Train ML Models for Parkinson's Disease Detection
Creates synthetic realistic training data and trains ensemble models
"""

import numpy as np
import os
import sys
import logging
from ml_models import ParkinsonMLPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_audio_features(n_samples=500, affected_ratio=0.5):
    """
    Generate synthetic audio features based on Parkinson's research
    
    Parkinson's patients typically show:
    - Higher pitch variability (jitter)
    - Lower harmonic-to-noise ratio
    - More spectral noise
    - Irregular prosody
    """
    n_affected = int(n_samples * affected_ratio)
    n_healthy = n_samples - n_affected
    
    # Assume we have around 150 audio features (from audio_features.py)
    n_features = 150
    
    # Generate healthy voice features
    healthy_features = np.random.randn(n_healthy, n_features)
    healthy_features[:, :50] *= 0.5  # Lower variability in MFCCs
    healthy_features[:, 50:70] += 0.8  # Higher spectral clarity
    healthy_features[:, 70:90] *= 0.6  # Stable prosody
    
    # Generate affected voice features
    affected_features = np.random.randn(n_affected, n_features)
    affected_features[:, :50] *= 1.5  # Higher variability in MFCCs
    affected_features[:, 50:70] -= 0.5  # Lower spectral clarity
    affected_features[:, 70:90] *= 1.4  # Irregular prosody
    
    # Add specific Parkinson's markers
    # Jitter (pitch variability) - higher in affected
    affected_features[:, 90] += np.random.uniform(0.5, 2.0, n_affected)
    healthy_features[:, 90] += np.random.uniform(0.0, 0.5, n_healthy)
    
    # HNR (Harmonic-to-Noise Ratio) - lower in affected
    affected_features[:, 91] -= np.random.uniform(3.0, 8.0, n_affected)
    healthy_features[:, 91] += np.random.uniform(2.0, 6.0, n_healthy)
    
    # Combine
    X = np.vstack([healthy_features, affected_features])
    y = np.array([0] * n_healthy + [1] * n_affected)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    logger.info(f"Generated {n_samples} synthetic audio feature samples")
    logger.info(f"  - Healthy: {n_healthy} samples")
    logger.info(f"  - Affected: {n_affected} samples")
    
    return X, y


def generate_synthetic_tremor_features(n_samples=500, affected_ratio=0.5):
    """
    Generate synthetic tremor features based on Parkinson's research
    
    Parkinson's patients typically show:
    - 4-6 Hz resting tremor
    - Higher motion variability
    - Reduced postural stability
    - Increased tremor amplitude
    """
    n_affected = int(n_samples * affected_ratio)
    n_healthy = n_samples - n_affected
    
    # Assume we have around 200 tremor features (from tremor_features.py)
    n_features = 200
    
    # Generate healthy motion features
    healthy_features = np.random.randn(n_healthy, n_features)
    healthy_features *= 0.3  # Lower overall variability
    
    # Generate affected motion features
    affected_features = np.random.randn(n_affected, n_features)
    affected_features *= 0.8  # Higher overall variability
    
    # Add specific tremor markers
    # Dominant frequency - 4-6 Hz in affected, varied in healthy
    affected_features[:, 0] = np.random.uniform(4.0, 6.0, n_affected)  # Tremor frequency
    healthy_features[:, 0] = np.random.uniform(0.5, 3.5, n_healthy)
    
    # Tremor band power - higher in affected
    affected_features[:, 10:14] += np.random.uniform(2.0, 5.0, (n_affected, 4))
    healthy_features[:, 10:14] += np.random.uniform(0.0, 1.0, (n_healthy, 4))
    
    # Stability index - higher (less stable) in affected
    affected_features[:, 150] += np.random.uniform(0.5, 1.5, n_affected)
    healthy_features[:, 150] += np.random.uniform(0.0, 0.3, n_healthy)
    
    # Motion variability - higher in affected
    affected_features[:, 30:50] *= 2.0
    healthy_features[:, 30:50] *= 0.5
    
    # Tremor amplitude - higher in affected
    affected_features[:, 100:110] += np.random.uniform(1.0, 3.0, (n_affected, 10))
    healthy_features[:, 100:110] += np.random.uniform(0.0, 0.5, (n_healthy, 10))
    
    # Combine
    X = np.vstack([healthy_features, affected_features])
    y = np.array([0] * n_healthy + [1] * n_affected)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    logger.info(f"Generated {n_samples} synthetic tremor feature samples")
    logger.info(f"  - Healthy: {n_healthy} samples")
    logger.info(f"  - Affected: {n_affected} samples")
    
    return X, y


def train_models(n_samples=1000):
    """
    Train the Parkinson's detection models
    
    Args:
        n_samples: Number of training samples to generate
    """
    logger.info("="*60)
    logger.info("Starting Parkinson's ML Model Training")
    logger.info("="*60)
    
    # Generate synthetic training data
    logger.info("\nGenerating synthetic training data...")
    X_voice, y_voice = generate_synthetic_audio_features(n_samples)
    X_tremor, y_tremor = generate_synthetic_tremor_features(n_samples)
    
    # Initialize pipeline
    logger.info("\nInitializing ML Pipeline...")
    pipeline = ParkinsonMLPipeline(model_dir='models')
    
    # Train models
    logger.info("\nTraining models (this may take a few minutes)...")
    pipeline.train_models(X_voice, y_voice, X_tremor, y_tremor)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info("\nModels saved to 'models/' directory")
    logger.info("You can now use the trained models for prediction")
    
    return pipeline


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Parkinson\'s Detection Models')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of training samples to generate (default: 1000)')
    
    args = parser.parse_args()
    
    try:
        # Train models
        pipeline = train_models(n_samples=args.samples)
        
        logger.info("\n" + "="*60)
        logger.info("Next Steps:")
        logger.info("="*60)
        logger.info("1. Start the backend server: python app.py")
        logger.info("2. The models will be automatically loaded")
        logger.info("3. Begin testing with real audio and motion data")
        logger.info("\nNote: These are trained on synthetic data.")
        logger.info("For production use, train on real clinical data.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
