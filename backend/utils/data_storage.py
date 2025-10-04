"""
Data Storage Manager
Stores recorded voice samples and tremor data with proper labeling
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

class DataStorageManager:
    def __init__(self, base_path='recorded_data'):
        self.base_path = Path(base_path)
        self.voice_path = self.base_path / 'voice_recordings'
        self.tremor_path = self.base_path / 'tremor_data'
        self.metadata_path = self.base_path / 'metadata'
        
        # Create directories
        self.voice_path.mkdir(parents=True, exist_ok=True)
        self.tremor_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metadata
        self.metadata_file = self.metadata_path / 'recordings_metadata.json'
        self.tremor_csv = self.tremor_path / 'recorded_tremor_data.csv'
        self.load_metadata()
    
    def load_metadata(self):
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'voice_recordings': [],
                'tremor_recordings': [],
                'total_recordings': 0
            }
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def store_voice_recording(self, audio_file_path, prediction, confidence, 
                             voice_confidence, features, user_label=None):
        """
        Store voice recording with metadata
        
        Args:
            audio_file_path: Path to the audio file
            prediction: Model prediction (0=healthy, 1=parkinson's)
            confidence: Overall confidence score
            voice_confidence: Voice-specific confidence
            features: Extracted features dictionary
            user_label: Optional user-provided label for verification
        """
        # Validate audio file path
        if audio_file_path is None:
            raise ValueError("audio_file_path cannot be None")
        
        if not os.path.exists(audio_file_path):
            raise ValueError(f"Audio file does not exist: {audio_file_path}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        recording_id = f"voice_{timestamp}"
        
        # Determine label based on prediction
        label = 'parkinsons' if prediction == 1 else 'healthy'
        label_folder = self.voice_path / label
        label_folder.mkdir(exist_ok=True)
        
        # Copy audio file to storage
        file_extension = Path(audio_file_path).suffix
        new_file_path = label_folder / f"{recording_id}{file_extension}"
        shutil.copy2(audio_file_path, new_file_path)
        
        # Create metadata entry
        metadata_entry = {
            'recording_id': recording_id,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(new_file_path),
            'predicted_label': label,
            'prediction_value': int(prediction),
            'confidence': float(confidence),
            'voice_confidence': float(voice_confidence),
            'user_label': user_label,
            'features': {k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in features.items()}
        }
        
        # Add to metadata
        self.metadata['voice_recordings'].append(metadata_entry)
        self.metadata['total_recordings'] += 1
        self.save_metadata()
        
        logger.info(f"Stored voice recording: {recording_id} (label: {label})")
        return recording_id, str(new_file_path)
    
    def store_tremor_data(self, motion_data, prediction, confidence, 
                         tremor_confidence, features, user_label=None):
        """
        Store tremor data with metadata in CSV format matching dataset structure
        
        Args:
            motion_data: Raw motion sensor data
            prediction: Model prediction (0=healthy, 1=affected)
            confidence: Overall confidence score
            tremor_confidence: Tremor-specific confidence
            features: Extracted features dictionary
            user_label: Optional user-provided label
        """
        # Validate motion data
        if motion_data is None or len(motion_data) == 0:
            raise ValueError("motion_data cannot be None or empty")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        recording_id = f"tremor_{timestamp}"
        
        # Prepare data for CSV
        tremor_record = {
            'recording_id': recording_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'tremor_confidence': float(tremor_confidence),
            'user_label': user_label,
            'num_samples': len(motion_data) if motion_data else 0,
        }
        
        # Add extracted features
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                tremor_record[feature_name] = float(feature_value)
        
        # Add tremor type labels (matching CSV format)
        tremor_record['Rest_tremor'] = 1 if prediction == 1 else 0
        tremor_record['Postural_tremor'] = 0  # Could be enhanced later
        tremor_record['Kinetic_tremor'] = 0   # Could be enhanced later
        
        # Save to CSV
        df_new = pd.DataFrame([tremor_record])
        
        if self.tremor_csv.exists():
            # Append to existing CSV
            df_existing = pd.read_csv(self.tremor_csv)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(self.tremor_csv, index=False)
        else:
            # Create new CSV
            df_new.to_csv(self.tremor_csv, index=False)
        
        # Also store in JSON metadata
        metadata_entry = {
            'recording_id': recording_id,
            'timestamp': datetime.now().isoformat(),
            'csv_file': str(self.tremor_csv),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'tremor_confidence': float(tremor_confidence),
            'user_label': user_label
        }
        
        self.metadata['tremor_recordings'].append(metadata_entry)
        self.metadata['total_recordings'] += 1
        self.save_metadata()
        
        logger.info(f"Stored tremor data: {recording_id}")
        return recording_id, str(self.tremor_csv)
    
    def store_combined_analysis(self, voice_recording_id, tremor_recording_id,
                                combined_prediction, combined_confidence):
        """Store combined analysis results"""
        combined_entry = {
            'timestamp': datetime.now().isoformat(),
            'voice_recording_id': voice_recording_id,
            'tremor_recording_id': tremor_recording_id,
            'combined_prediction': combined_prediction,
            'combined_confidence': float(combined_confidence)
        }
        
        if 'combined_analyses' not in self.metadata:
            self.metadata['combined_analyses'] = []
        
        self.metadata['combined_analyses'].append(combined_entry)
        self.save_metadata()
        
        return combined_entry
    
    def get_statistics(self):
        """Get storage statistics"""
        stats = {
            'total_recordings': self.metadata.get('total_recordings', 0),
            'voice_recordings': len(self.metadata.get('voice_recordings', [])),
            'tremor_recordings': len(self.metadata.get('tremor_recordings', [])),
            'combined_analyses': len(self.metadata.get('combined_analyses', []))
        }
        
        # Count by label
        voice_recs = self.metadata.get('voice_recordings', [])
        stats['voice_healthy'] = sum(1 for r in voice_recs if r['predicted_label'] == 'healthy')
        stats['voice_parkinsons'] = sum(1 for r in voice_recs if r['predicted_label'] == 'parkinsons')
        
        tremor_recs = self.metadata.get('tremor_recordings', [])
        stats['tremor_healthy'] = sum(1 for r in tremor_recs if r['prediction'] == 0)
        stats['tremor_affected'] = sum(1 for r in tremor_recs if r['prediction'] == 1)
        
        return stats
    
    def get_recent_recordings(self, count=10, recording_type='all'):
        """Get recent recordings"""
        if recording_type == 'voice':
            recordings = self.metadata.get('voice_recordings', [])
        elif recording_type == 'tremor':
            recordings = self.metadata.get('tremor_recordings', [])
        elif recording_type == 'combined':
            recordings = self.metadata.get('combined_analyses', [])
        else:
            recordings = (self.metadata.get('voice_recordings', []) + 
                         self.metadata.get('tremor_recordings', []))
        
        # Sort by timestamp (descending)
        sorted_recordings = sorted(recordings, 
                                   key=lambda x: x['timestamp'], 
                                   reverse=True)
        
        return sorted_recordings[:count]


if __name__ == "__main__":
    # Test the storage manager
    logging.basicConfig(level=logging.INFO)
    
    manager = DataStorageManager()
    
    print("\n=== Storage Statistics ===")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Recent Recordings ===")
    recent = manager.get_recent_recordings(count=5)
    for rec in recent:
        print(f"{rec.get('recording_id', 'N/A')} - {rec['timestamp']}")
