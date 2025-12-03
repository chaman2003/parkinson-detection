"""
Test Audio Files Directly from Dataset
Run this script to verify that the ML model correctly classifies
healthy vs Parkinson's audio files from the dataset.
"""

import os
import sys
import random

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ml_models import ParkinsonMLPipeline
from utils.audio_features_optimized import OptimizedAudioExtractor

def test_audio_files():
    """Test audio files directly from the dataset"""
    
    # Initialize the pipeline
    print("="*70)
    print("AUDIO FILE CLASSIFICATION TEST")
    print("="*70)
    
    pipeline = ParkinsonMLPipeline()
    extractor = OptimizedAudioExtractor()
    
    # Check if models are loaded
    if pipeline.voice_model is None:
        print("ERROR: Voice model not loaded!")
        print("Run train.py first to train the models")
        return
    
    print(f"✓ Voice model loaded successfully")
    print(f"✓ Voice feature names: {len(pipeline.voice_feature_names)} features")
    
    # Dataset directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    healthy_dir = os.path.join(base_dir, "datasets", "voice_dataset", "healthy")
    parkinson_dir = os.path.join(base_dir, "datasets", "voice_dataset", "parkinson")
    
    # Get audio files
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.wav')]
    parkinson_files = [os.path.join(parkinson_dir, f) for f in os.listdir(parkinson_dir) if f.endswith('.wav')]
    
    print(f"\nFound {len(healthy_files)} healthy files")
    print(f"Found {len(parkinson_files)} Parkinson's files")
    
    # Sample files for testing - use more files for better accuracy measure
    test_healthy = random.sample(healthy_files, min(15, len(healthy_files)))
    test_parkinson = random.sample(parkinson_files, min(15, len(parkinson_files)))
    
    print("\n" + "="*70)
    print("TESTING HEALTHY FILES (should show 'Not Affected' with high confidence)")
    print("="*70)
    
    healthy_correct = 0
    for audio_path in test_healthy:
        filename = os.path.basename(audio_path)
        print(f"\nTesting: {filename}")
        
        # Extract features
        features = extractor.extract_features_fast(audio_path)
        
        # Check if silence was detected
        if features.get('_silence_detected', False):
            print(f"  ⚠️ SILENCE DETECTED - Features may be default values")
            silence_metrics = features.get('_silence_metrics', {})
            print(f"     RMS dB: {silence_metrics.get('rms_db', 'N/A')}")
            print(f"     Energy: {silence_metrics.get('energy', 'N/A')}")
            print(f"     Voiced %: {silence_metrics.get('voiced_percent', 'N/A')}")
        
        # Create fake motion data (minimal)
        fake_motion = [{'timestamp': i, 'accelerationX': 0.0, 'accelerationY': 0.0, 'accelerationZ': 9.8} for i in range(100)]
        
        # Run analysis
        results = pipeline.analyze(audio_path, fake_motion)
        
        prediction = results.get('prediction', 'Unknown')
        voice_conf = results.get('voice_confidence', 0)
        
        is_correct = prediction == "Not Affected"
        if is_correct:
            healthy_correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"  {status} Prediction: {prediction}")
        print(f"     Voice Confidence: {voice_conf:.1f}%")
    
    print("\n" + "="*70)
    print("TESTING PARKINSON'S FILES (should show 'Affected' with high confidence)")
    print("="*70)
    
    parkinson_correct = 0
    for audio_path in test_parkinson:
        filename = os.path.basename(audio_path)
        print(f"\nTesting: {filename}")
        
        # Extract features
        features = extractor.extract_features_fast(audio_path)
        
        # Check if silence was detected
        if features.get('_silence_detected', False):
            print(f"  ⚠️ SILENCE DETECTED - Features may be default values")
            silence_metrics = features.get('_silence_metrics', {})
            print(f"     RMS dB: {silence_metrics.get('rms_db', 'N/A')}")
            print(f"     Energy: {silence_metrics.get('energy', 'N/A')}")
            print(f"     Voiced %: {silence_metrics.get('voiced_percent', 'N/A')}")
        
        # Create fake motion data (minimal)
        fake_motion = [{'timestamp': i, 'accelerationX': 0.0, 'accelerationY': 0.0, 'accelerationZ': 9.8} for i in range(100)]
        
        # Run analysis
        results = pipeline.analyze(audio_path, fake_motion)
        
        prediction = results.get('prediction', 'Unknown')
        voice_conf = results.get('voice_confidence', 0)
        
        is_correct = prediction == "Affected"
        if is_correct:
            parkinson_correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"  {status} Prediction: {prediction}")
        print(f"     Voice Confidence: {voice_conf:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    healthy_acc = (healthy_correct / len(test_healthy)) * 100 if test_healthy else 0
    parkinson_acc = (parkinson_correct / len(test_parkinson)) * 100 if test_parkinson else 0
    total_correct = healthy_correct + parkinson_correct
    total_files = len(test_healthy) + len(test_parkinson)
    overall_acc = (total_correct / total_files) * 100 if total_files else 0
    
    print(f"Healthy Classification:    {healthy_correct}/{len(test_healthy)} ({healthy_acc:.1f}%)")
    print(f"Parkinson's Classification: {parkinson_correct}/{len(test_parkinson)} ({parkinson_acc:.1f}%)")
    print(f"Overall Accuracy:          {total_correct}/{total_files} ({overall_acc:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    test_audio_files()
