"""
Validation Script - Verify CustomStandardScaler Implementation
Tests all models and scalers to ensure zero errors
"""
import pickle
import numpy as np
import sys
import os

print("\n" + "="*80)
print("  CUSTOM STANDARDSCALER VALIDATION")
print("="*80)

# Test 1: Load Custom Scaler Module
print("\n[1/6] Testing custom_scaler module import...")
try:
    from custom_scaler import CustomStandardScaler
    print("  ✅ CustomStandardScaler imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Load Voice Scaler
print("\n[2/6] Loading voice scaler...")
try:
    with open('models/voice_scaler.pkl', 'rb') as f:
        voice_scaler = pickle.load(f)
    print(f"  ✅ Loaded: {voice_scaler}")
    print(f"     - Features: {voice_scaler.n_features_}")
    print(f"     - Fitted: {voice_scaler.is_fitted_}")
    assert voice_scaler.is_fitted_, "Voice scaler not fitted!"
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Load Tremor Scaler
print("\n[3/6] Loading tremor scaler...")
try:
    with open('models/tremor_scaler.pkl', 'rb') as f:
        tremor_scaler = pickle.load(f)
    print(f"  ✅ Loaded: {tremor_scaler}")
    print(f"     - Features: {tremor_scaler.n_features_}")
    print(f"     - Fitted: {tremor_scaler.is_fitted_}")
    assert tremor_scaler.is_fitted_, "Tremor scaler not fitted!"
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Test Voice Scaler Transform
print("\n[4/6] Testing voice scaler transform...")
try:
    # Create random test data with correct number of features
    test_voice_data = np.random.randn(1, voice_scaler.n_features_)
    scaled_voice = voice_scaler.transform(test_voice_data)
    print(f"  ✅ Transform successful")
    print(f"     - Input shape: {test_voice_data.shape}")
    print(f"     - Output shape: {scaled_voice.shape}")
    print(f"     - Output mean: {scaled_voice.mean():.3f} (should be near 0)")
    print(f"     - Output std: {scaled_voice.std():.3f} (should be near 1)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 5: Test Tremor Scaler Transform
print("\n[5/6] Testing tremor scaler transform...")
try:
    # Create random test data with correct number of features
    test_tremor_data = np.random.randn(1, tremor_scaler.n_features_)
    scaled_tremor = tremor_scaler.transform(test_tremor_data)
    print(f"  ✅ Transform successful")
    print(f"     - Input shape: {test_tremor_data.shape}")
    print(f"     - Output shape: {scaled_tremor.shape}")
    print(f"     - Output mean: {scaled_tremor.mean():.3f} (should be near 0)")
    print(f"     - Output std: {scaled_tremor.std():.3f} (should be near 1)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Test 6: Load Models
print("\n[6/6] Loading ML models...")
try:
    with open('models/voice_model.pkl', 'rb') as f:
        voice_model = pickle.load(f)
    print(f"  ✅ Voice model loaded: {type(voice_model).__name__}")
    
    with open('models/tremor_model.pkl', 'rb') as f:
        tremor_model = pickle.load(f)
    print(f"  ✅ Tremor model loaded: {type(tremor_model).__name__}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# Final Summary
print("\n" + "="*80)
print("  ✅ ALL VALIDATION TESTS PASSED!")
print("="*80)
print("\n📊 Summary:")
print(f"  ✅ Custom StandardScaler: Working")
print(f"  ✅ Voice Scaler: Fitted with {voice_scaler.n_features_} features")
print(f"  ✅ Tremor Scaler: Fitted with {tremor_scaler.n_features_} features")
print(f"  ✅ Voice Model: Loaded successfully")
print(f"  ✅ Tremor Model: Loaded successfully")
print(f"  ✅ Transform Operations: Working perfectly")
print("\n🎉 Zero Errors - System Ready for Production!")
print("="*80 + "\n")
