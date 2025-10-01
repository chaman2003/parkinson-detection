"""
Data Loader for Real Parkinson's Datasets
Loads voice samples from Healthy_AH and Parkinsons_AH folders
Loads tremor data from parkinson_tremor_dataset.csv
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, base_path='datasets'):
        self.base_path = Path(base_path)
        self.voice_path = self.base_path / 'voice_dataset'
        self.tremor_csv = self.base_path / 'parkinson_tremor_dataset.csv'
        
        # Voice dataset paths
        self.healthy_voice_path = self.voice_path / 'Healthy_AH'
        self.parkinsons_voice_path = self.voice_path / 'Parkinsons_AH'
        
    def load_voice_dataset(self):
        """Load all voice samples from both healthy and Parkinson's folders"""
        logger.info("Loading voice dataset...")
        
        voice_files = []
        labels = []
        
        # Load healthy samples (label = 0)
        if self.healthy_voice_path.exists():
            healthy_files = list(self.healthy_voice_path.glob('*.wav'))
            logger.info(f"Found {len(healthy_files)} healthy voice samples")
            voice_files.extend(healthy_files)
            labels.extend([0] * len(healthy_files))
        else:
            logger.warning(f"Healthy voice path not found: {self.healthy_voice_path}")
        
        # Load Parkinson's samples (label = 1)
        if self.parkinsons_voice_path.exists():
            parkinsons_files = list(self.parkinsons_voice_path.glob('*.wav'))
            logger.info(f"Found {len(parkinsons_files)} Parkinson's voice samples")
            voice_files.extend(parkinsons_files)
            labels.extend([1] * len(parkinsons_files))
        else:
            logger.warning(f"Parkinsons voice path not found: {self.parkinsons_voice_path}")
        
        return voice_files, np.array(labels)
    
    def load_tremor_dataset(self):
        """Load tremor data from CSV file"""
        logger.info("Loading tremor dataset...")
        
        if not self.tremor_csv.exists():
            raise FileNotFoundError(f"Tremor dataset not found: {self.tremor_csv}")
        
        # Read CSV file
        df = pd.read_csv(self.tremor_csv)
        logger.info(f"Loaded tremor dataset with {len(df)} samples and {len(df.columns)} features")
        
        # Extract feature columns (all except the label columns at the end)
        label_columns = ['Rest_tremor', 'Postural_tremor', 'Kinetic_tremor']
        feature_columns = [col for col in df.columns if col not in label_columns and 
                          col not in ['subject_id', 'start_timestamp', 'end_timestamp']]
        
        features = df[feature_columns].values
        
        # Create labels: 1 if any tremor type is present, 0 otherwise
        labels = ((df['Rest_tremor'] == 1) | 
                 (df['Postural_tremor'] == 1) | 
                 (df['Kinetic_tremor'] == 1)).astype(int).values
        
        logger.info(f"Tremor features shape: {features.shape}")
        logger.info(f"Tremor labels - Healthy: {np.sum(labels == 0)}, Affected: {np.sum(labels == 1)}")
        
        return features, labels, df
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        info = {
            'voice': {
                'healthy_path': str(self.healthy_voice_path),
                'parkinsons_path': str(self.parkinsons_voice_path),
                'healthy_exists': self.healthy_voice_path.exists(),
                'parkinsons_exists': self.parkinsons_voice_path.exists(),
            },
            'tremor': {
                'csv_path': str(self.tremor_csv),
                'exists': self.tremor_csv.exists(),
            }
        }
        
        if self.healthy_voice_path.exists():
            info['voice']['healthy_count'] = len(list(self.healthy_voice_path.glob('*.wav')))
        
        if self.parkinsons_voice_path.exists():
            info['voice']['parkinsons_count'] = len(list(self.parkinsons_voice_path.glob('*.wav')))
        
        if self.tremor_csv.exists():
            df = pd.read_csv(self.tremor_csv)
            info['tremor']['sample_count'] = len(df)
            info['tremor']['feature_count'] = len([col for col in df.columns 
                                                   if col not in ['subject_id', 'start_timestamp', 
                                                                 'end_timestamp', 'Rest_tremor', 
                                                                 'Postural_tremor', 'Kinetic_tremor']])
        
        return info


def load_single_voice_file(file_path, target_sr=22050):
    """Load a single voice file and return audio data"""
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        return y, sr
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = DatasetLoader()
    
    print("\n=== Dataset Information ===")
    info = loader.get_dataset_info()
    print(f"\nVoice Dataset:")
    print(f"  Healthy samples: {info['voice'].get('healthy_count', 'N/A')}")
    print(f"  Parkinson's samples: {info['voice'].get('parkinsons_count', 'N/A')}")
    print(f"\nTremor Dataset:")
    print(f"  Total samples: {info['tremor'].get('sample_count', 'N/A')}")
    print(f"  Features per sample: {info['tremor'].get('feature_count', 'N/A')}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    voice_files, voice_labels = loader.load_voice_dataset()
    print(f"Loaded {len(voice_files)} voice samples")
    print(f"  Healthy: {np.sum(voice_labels == 0)}")
    print(f"  Parkinson's: {np.sum(voice_labels == 1)}")
    
    tremor_features, tremor_labels, tremor_df = loader.load_tremor_dataset()
    print(f"\nLoaded {len(tremor_features)} tremor samples")
    print(f"  Healthy: {np.sum(tremor_labels == 0)}")
    print(f"  Affected: {np.sum(tremor_labels == 1)}")
    print(f"  Feature dimensions: {tremor_features.shape[1]}")
