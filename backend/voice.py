#!/usr/bin/env python
"""
Voice-based Parkinson's Detection CLI Tool
Test voice recordings from files or microphone
Usage: python voice.py [audio_file_path]
"""

import sys
import os
import argparse
import numpy as np
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_features import AudioFeatureExtractor
from ml_models import ParkinsonMLPipeline
from data_storage import DataStorageManager
from data_loader import load_single_voice_file

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class VoiceTester:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.ml_pipeline = ParkinsonMLPipeline()
        self.storage_manager = DataStorageManager()
        
        # Load models
        try:
            self.ml_pipeline.load_models('models')
            logger.info("✓ Models loaded successfully\n")
        except Exception as e:
            logger.error(f"✗ Error loading models: {e}")
            logger.error("Please train models first by running: python app.py\n")
            sys.exit(1)
    
    def test_voice_file(self, file_path, save_result=True, user_label=None):
        """Test a single voice file"""
        logger.info("=" * 70)
        logger.info(f"Testing Voice File: {Path(file_path).name}")
        logger.info("=" * 70)
        
        # Load audio file
        logger.info("Loading audio file...")
        y, sr = load_single_voice_file(file_path)
        
        if y is None:
            logger.error("✗ Failed to load audio file")
            return None
        
        logger.info(f"✓ Audio loaded: {len(y)} samples at {sr} Hz ({len(y)/sr:.2f} seconds)")
        
        # Extract features
        logger.info("\nExtracting audio features...")
        try:
            features = self.feature_extractor.extract_all_features(y, sr)
            logger.info(f"✓ Extracted {len(features)} features")
        except Exception as e:
            logger.error(f"✗ Feature extraction failed: {e}")
            return None
        
        # Prepare features for prediction
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Make prediction
        logger.info("\nRunning ML analysis...")
        try:
            result = self.ml_pipeline.predict_voice(feature_vector)
            
            prediction = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            logger.info("✓ Analysis complete\n")
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {e}")
            return None
        
        # Display results
        logger.info("=" * 70)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 70)
        
        diagnosis = "PARKINSON'S AFFECTED" if prediction == 1 else "HEALTHY"
        logger.info(f"\nPrediction: {diagnosis}")
        logger.info(f"Confidence: {confidence:.1%}")
        logger.info(f"\nProbabilities:")
        logger.info(f"  Healthy:     {probabilities[0]:.1%}")
        logger.info(f"  Parkinson's: {probabilities[1]:.1%}")
        
        # Display key features
        logger.info(f"\nKey Voice Features:")
        key_features = ['jitter', 'shimmer', 'hnr', 'mfcc_mean_0', 'spectral_centroid_mean']
        for feat_name in key_features:
            if feat_name in features:
                logger.info(f"  {feat_name}: {features[feat_name]:.4f}")
        
        # Save result if requested
        if save_result:
            logger.info(f"\nSaving result...")
            recording_id, stored_path = self.storage_manager.store_voice_recording(
                audio_file_path=file_path,
                prediction=prediction,
                confidence=confidence,
                voice_confidence=confidence,
                features=features,
                user_label=user_label
            )
            logger.info(f"✓ Saved as: {recording_id}")
            logger.info(f"  Location: {stored_path}")
        
        logger.info("\n" + "=" * 70 + "\n")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'features': features
        }
    
    def batch_test_directory(self, directory_path, pattern='*.wav'):
        """Test all audio files in a directory"""
        directory = Path(directory_path)
        audio_files = list(directory.glob(pattern))
        
        if not audio_files:
            logger.error(f"No audio files found in {directory_path}")
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH TESTING: {len(audio_files)} files")
        logger.info(f"{'='*70}\n")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] Testing: {audio_file.name}")
            result = self.test_voice_file(str(audio_file), save_result=False)
            if result:
                results.append({
                    'file': audio_file.name,
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                })
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("BATCH TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\nTotal files tested: {len(results)}")
        
        healthy_count = sum(1 for r in results if r['prediction'] == 0)
        parkinsons_count = sum(1 for r in results if r['prediction'] == 1)
        
        logger.info(f"  Healthy: {healthy_count}")
        logger.info(f"  Parkinson's: {parkinsons_count}")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        logger.info(f"\nAverage confidence: {avg_confidence:.1%}\n")
        
        return results
    
    def test_dataset_samples(self, num_samples=5):
        """Test random samples from the training dataset"""
        from data_loader import DatasetLoader
        
        loader = DatasetLoader()
        voice_files, labels = loader.load_voice_dataset()
        
        if len(voice_files) == 0:
            logger.error("No voice files found in dataset")
            return
        
        # Random selection
        indices = np.random.choice(len(voice_files), min(num_samples, len(voice_files)), replace=False)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING DATASET SAMPLES: {len(indices)} random files")
        logger.info(f"{'='*70}\n")
        
        correct = 0
        for i, idx in enumerate(indices, 1):
            file_path = voice_files[idx]
            true_label = labels[idx]
            true_label_str = "Parkinson's" if true_label == 1 else "Healthy"
            
            logger.info(f"\n[{i}/{len(indices)}] File: {file_path.name}")
            logger.info(f"True label: {true_label_str}")
            
            result = self.test_voice_file(str(file_path), save_result=False)
            
            if result and result['prediction'] == true_label:
                correct += 1
                logger.info("✓ CORRECT")
            else:
                logger.info("✗ INCORRECT")
        
        # Accuracy
        accuracy = (correct / len(indices)) * 100
        logger.info(f"\n{'='*70}")
        logger.info(f"Sample Test Accuracy: {accuracy:.1f}% ({correct}/{len(indices)})")
        logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Voice-based Parkinson\'s Detection CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice.py recording.wav                    # Test single file
  python voice.py recording.wav --label healthy    # Test with known label
  python voice.py --directory ./samples            # Test all files in directory
  python voice.py --test-dataset 10                # Test 10 random dataset samples
        """
    )
    
    parser.add_argument('file', nargs='?', help='Audio file to test')
    parser.add_argument('--directory', '-d', help='Test all audio files in directory')
    parser.add_argument('--pattern', default='*.wav', help='File pattern for directory (default: *.wav)')
    parser.add_argument('--label', choices=['healthy', 'parkinsons'], help='Known label for verification')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    parser.add_argument('--test-dataset', type=int, metavar='N', help='Test N random samples from dataset')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = VoiceTester()
    
    # Test dataset samples
    if args.test_dataset:
        tester.test_dataset_samples(num_samples=args.test_dataset)
        return
    
    # Test directory
    if args.directory:
        tester.batch_test_directory(args.directory, pattern=args.pattern)
        return
    
    # Test single file
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        
        user_label = 1 if args.label == 'parkinsons' else 0 if args.label == 'healthy' else None
        save_result = not args.no_save
        
        tester.test_voice_file(args.file, save_result=save_result, user_label=user_label)
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
