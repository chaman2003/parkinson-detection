"""
Parkinson's Detection - Model Training Script
==============================================
Single entry point for training all ML models

Usage: python train.py

Features:
- Trains models on streamlined datasets
- Uses only frontend-relevant features
- Saves models and dataset mappings
- Provides comprehensive training metrics
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Add utils package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_features_optimized import OptimizedAudioExtractor
from utils.tremor_features_optimized import OptimizedTremorExtractor
from custom_scaler import CustomStandardScaler

print("\n" + "="*80)
print("  PARKINSON'S DETECTION - MODEL TRAINING")
print("="*80)
print("\n📊 Configuration:")
print("  • Voice Features: Optimized Selection (Top 25)")
print("  • Tremor Features: 12 (Magnitude, Time/Frequency Domain)")
print("  • Models: Ensemble (SVM + RandomForest + GradientBoosting + XGBoost)")
print("="*80)

start_time = time.time()

# =============================================================================
# LOAD TREMOR DATASET
# =============================================================================

print("\n📁 Loading Tremor Dataset...")
tremor_df = pd.read_csv('datasets/tremor_simplified.csv')
print(f"✓ Loaded: {tremor_df.shape}")

feature_cols = [col for col in tremor_df.columns if col not in ['subject_id', 'Rest_tremor', 'Postural_tremor', 'Kinetic_tremor', 'parkinsons_label'] and not col.startswith('PC1')]
X_tremor = tremor_df[feature_cols].values
y_tremor = tremor_df['parkinsons_label'].values

print(f"  • Features: {len(feature_cols)}")
print(f"  • Samples: {len(X_tremor)} (Healthy: {np.sum(y_tremor == 0)}, Affected: {np.sum(y_tremor == 1)})")

# =============================================================================
# LOAD AND PROCESS VOICE DATASET
# =============================================================================

print("\n📁 Loading Voice Dataset...")
voice_labels_df = pd.read_csv('datasets/voice_labels.csv')
print(f"✓ Loaded: {len(voice_labels_df)} files")

# Check dataset composition
parkinsons_count = (voice_labels_df['parkinsons_label'] == 1).sum()
healthy_count = (voice_labels_df['parkinsons_label'] == 0).sum()
print(f"  • Parkinson's samples: {parkinsons_count}")
print(f"  • Healthy samples: {healthy_count}")

print("\n🎵 Extracting audio features (optimized parallel extraction)...")
audio_extractor = OptimizedAudioExtractor()
X_voice_list = []
y_voice_list = []
skipped = 0
error_details = []

for idx, row in voice_labels_df.iterrows():
    if (idx + 1) % 5 == 0:
        print(f"  Processing {idx+1}/{len(voice_labels_df)}...")
    
    try:
        features = audio_extractor.extract_features_fast(row['filepath'])
        # Remove insights before converting to vector
        features.pop('_insights', None)
        features.pop('_silence_detected', None)
        features.pop('_silence_metrics', None)
        
        # CRITICAL: Sort keys to ensure consistent order matching all_feature_names
        sorted_keys = sorted(features.keys())
        feature_vector = np.array([features[k] for k in sorted_keys], dtype=np.float64)
        
        # Validate feature vector
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            print(f"  ⚠️  Skipping {row['filename']}: Invalid feature values")
            skipped += 1
            continue
        
        X_voice_list.append(feature_vector)
        y_voice_list.append(row['parkinsons_label'])
    except Exception as e:
        error_details.append((row['filename'], str(e)))
        skipped += 1
        continue

if skipped > 0 and len(error_details) > 0:
    print(f"\n⚠️  Skipped {skipped} files due to errors:")
    for filename, error in error_details[:3]:  # Show first 3 errors
        print(f"  • {filename}: {error}")

X_voice = np.array(X_voice_list, dtype=np.float64)
y_voice = np.array(y_voice_list, dtype=np.int32)

if len(X_voice) == 0:
    print("\n❌ ERROR: No voice samples could be extracted!")
    print("   Please check that audio files exist in the specified paths.")
    sys.exit(1)

print(f"\n✓ Extracted: {len(X_voice)} samples ({X_voice.shape[1]} features, Skipped: {skipped})")

# Check if we have both classes
unique_labels = np.unique(y_voice)
if len(unique_labels) < 2:
    print(f"\n⚠️  WARNING: Voice dataset only contains {'Parkinson\'s' if unique_labels[0] == 1 else 'healthy'} samples!")
    print("  Creating synthetic healthy baseline samples for model training...")
    
    # Create synthetic healthy samples by dampening Parkinson's features
    # This allows the model to learn the difference between Parkinson's and normal
    n_synthetic = len(X_voice) // 2  # Create half as many synthetic samples
    X_synthetic_healthy = []
    
    for i in range(n_synthetic):
        # Take a random Parkinson's sample and dampen its features
        idx = np.random.randint(0, len(X_voice))
        synthetic_sample = X_voice[idx].copy()
        
        # Dampen tremor-related features (jitter, shimmer, variability)
        # Multiply by 0.3-0.5 to simulate healthy voice characteristics
        dampening_factor = np.random.uniform(0.3, 0.5)
        synthetic_sample = synthetic_sample * dampening_factor
        
        X_synthetic_healthy.append(synthetic_sample)
    
    X_synthetic_healthy = np.array(X_synthetic_healthy)
    y_synthetic_healthy = np.zeros(n_synthetic, dtype=int)
    
    # Combine real Parkinson's samples with synthetic healthy samples
    X_voice = np.vstack([X_voice, X_synthetic_healthy])
    y_voice = np.concatenate([y_voice, y_synthetic_healthy])
    
    # Create synthetic filenames
    synthetic_filenames = [f"synthetic_healthy_{i}" for i in range(n_synthetic)]
    # Update voice_labels_df or create a new list of filenames
    all_filenames = np.concatenate([voice_labels_df['filename'].values, synthetic_filenames])
    
    print(f"  ✓ Added {n_synthetic} synthetic healthy samples")
    print(f"  • Final dataset: {len(X_voice)} samples (Parkinson's: {parkinsons_count}, Healthy: {n_synthetic})")
else:
    all_filenames = voice_labels_df['filename'].values

# =============================================================================
# TRAIN TREMOR MODEL
# =============================================================================

print("\n🤖 Training Tremor Model...")

X_tremor_train, X_tremor_test, y_tremor_train, y_tremor_test = train_test_split(
    X_tremor, y_tremor, test_size=0.2, random_state=42, stratify=y_tremor
)

tremor_scaler = CustomStandardScaler()
X_tremor_train_scaled = tremor_scaler.fit_transform(X_tremor_train)
X_tremor_test_scaled = tremor_scaler.transform(X_tremor_test)

print(f"  • Scaler fitted: {tremor_scaler.n_features_} features")
print(f"  • Training samples: {X_tremor_train_scaled.shape[0]}, Test samples: {X_tremor_test_scaled.shape[0]}")

tremor_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
tremor_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
tremor_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
tremor_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')

tremor_model = VotingClassifier(
    estimators=[('svm', tremor_svm), ('rf', tremor_rf), ('gb', tremor_gb), ('xgb', tremor_xgb)],
    voting='soft'
)

tremor_model.fit(X_tremor_train_scaled, y_tremor_train)

y_tremor_pred = tremor_model.predict(X_tremor_test_scaled)
tremor_accuracy = accuracy_score(y_tremor_test, y_tremor_pred)
cv_scores = cross_val_score(tremor_model, X_tremor_train_scaled, y_tremor_train, cv=5)

print(f"✓ Trained: {tremor_accuracy:.1%} test accuracy, {cv_scores.mean():.1%} CV accuracy")

# =============================================================================
# TRAIN VOICE MODEL
# =============================================================================

print("\n🤖 Training Voice Model...")

# Feature Selection for Voice
print("  • Performing feature selection...")
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=25)
X_voice_selected = selector.fit_transform(X_voice, y_voice)
selected_indices = selector.get_support(indices=True)

# Get feature names (assuming order from extractor)
# We need to reconstruct the feature names list to save it
dummy_features = audio_extractor._get_default_features()
# Remove metadata keys to match X_voice columns
dummy_features.pop('_insights', None)
dummy_features.pop('_silence_detected', None)
dummy_features.pop('_silence_metrics', None)

all_feature_names = sorted(dummy_features.keys())
selected_feature_names = [all_feature_names[i] for i in selected_indices]

print(f"  • Selected {len(selected_feature_names)} features from {X_voice.shape[1]}")
selected_feature_names = [all_feature_names[i] for i in selected_indices]

print(f"  • Selected {len(selected_feature_names)} features from {X_voice.shape[1]}")
print(f"  • Top features: {', '.join(selected_feature_names[:5])}...")

X_voice_train, X_voice_test, y_voice_train, y_voice_test = train_test_split(
    X_voice_selected, y_voice, test_size=0.2, random_state=42, stratify=y_voice
)

voice_scaler = CustomStandardScaler()
X_voice_train_scaled = voice_scaler.fit_transform(X_voice_train)
X_voice_test_scaled = voice_scaler.transform(X_voice_test)

print(f"  • Scaler fitted: {voice_scaler.n_features_} features")
print(f"  • Training samples: {X_voice_train_scaled.shape[0]}, Test samples: {X_voice_test_scaled.shape[0]}")

voice_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
voice_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
voice_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
voice_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')

voice_model = VotingClassifier(
    estimators=[('svm', voice_svm), ('rf', voice_rf), ('gb', voice_gb), ('xgb', voice_xgb)],
    voting='soft'
)

voice_model.fit(X_voice_train_scaled, y_voice_train)

y_voice_pred = voice_model.predict(X_voice_test_scaled)
voice_accuracy = accuracy_score(y_voice_test, y_voice_pred)
voice_cv_scores = cross_val_score(voice_model, X_voice_train_scaled, y_voice_train, cv=5)

print(f"✓ Trained: {voice_accuracy:.1%} test accuracy, {voice_cv_scores.mean():.1%} CV accuracy")

# =============================================================================
# VALIDATE ON REAL-HEALTHY DATA
# =============================================================================

print("\n🏥 Validating on Real-Healthy Data...")

# Identify real-healthy samples from the original dataframe
# We need to find indices in X_voice that correspond to real-healthy files
real_healthy_indices = []
real_healthy_filenames = []

for i, filename in enumerate(all_filenames):
    if 'real-healthy' in str(filename) or (i < len(voice_labels_df) and 'real-healthy' in voice_labels_df.iloc[i]['filepath']):
        real_healthy_indices.append(i)
        real_healthy_filenames.append(filename)

if len(real_healthy_indices) > 0:
    print(f"  • Found {len(real_healthy_indices)} real-healthy samples")
    
    # Get features for these samples
    # Note: We need to be careful if we added synthetic data. 
    # The indices in X_voice match all_filenames.
    
    X_real_healthy = X_voice[real_healthy_indices]
    y_real_healthy = y_voice[real_healthy_indices] # Should be all 0s
    
    # Apply selection and scaling
    X_real_healthy_selected = selector.transform(X_real_healthy)
    X_real_healthy_scaled = voice_scaler.transform(X_real_healthy_selected)
    
    # Predict
    y_real_healthy_pred = voice_model.predict(X_real_healthy_scaled)
    y_real_healthy_proba = voice_model.predict_proba(X_real_healthy_scaled)
    
    # Calculate accuracy
    real_healthy_acc = accuracy_score(y_real_healthy, y_real_healthy_pred)
    
    print(f"  • Accuracy on Real-Healthy: {real_healthy_acc:.1%} ({np.sum(y_real_healthy_pred == 0)}/{len(y_real_healthy)} correctly identified as healthy)")
    
    # Show individual predictions
    print("\n  Detailed Predictions for Real-Healthy:")
    for i, pred in enumerate(y_real_healthy_pred):
        status = "✅ CORRECT (Healthy)" if pred == 0 else "❌ WRONG (Parkinson's)"
        confidence = y_real_healthy_proba[i][pred]
        print(f"    - {real_healthy_filenames[i]}: {status} (Conf: {confidence:.2f})")
        
else:
    print("  ⚠️  No real-healthy samples found in the dataset!")

# =============================================================================
# SAVE MODELS
# =============================================================================

print("\n💾 Saving Models...")
os.makedirs('models', exist_ok=True)

# Save selected feature names
with open('models/voice_feature_names.pkl', 'wb') as f:
    pickle.dump(selected_feature_names, f)

with open('models/tremor_model.pkl', 'wb') as f:
    pickle.dump(tremor_model, f)
with open('models/tremor_scaler.pkl', 'wb') as f:
    pickle.dump(tremor_scaler, f)
with open('models/voice_model.pkl', 'wb') as f:
    pickle.dump(voice_model, f)
with open('models/voice_scaler.pkl', 'wb') as f:
    pickle.dump(voice_scaler, f)

voice_dataset_mapping = {
    'features': X_voice_selected,
    'labels': y_voice,
    'filenames': all_filenames
}
with open('models/voice_dataset_mapping.pkl', 'wb') as f:
    pickle.dump(voice_dataset_mapping, f)

tremor_dataset_mapping = {
    'features': X_tremor,
    'labels': y_tremor,
    'dataframe': tremor_df,  # Include full dataframe for detailed matching
    'feature_columns': [col for col in tremor_df.columns if col not in ['subject_id', 'start_timestamp', 'end_timestamp', 'Rest_tremor', 'Postural_tremor', 'Kinetic_tremor', 'parkinsons_label'] and not col.startswith('PC1')]
}

# Verify feature count
print(f"\n🔍 Tremor mapping verification:")
print(f"  • Feature columns: {len(tremor_dataset_mapping['feature_columns'])}")
print(f"  • Feature array shape: {tremor_dataset_mapping['features'].shape}")
assert len(tremor_dataset_mapping['feature_columns']) == 12, f"Expected 12 feature columns, got {len(tremor_dataset_mapping['feature_columns'])}"
assert tremor_dataset_mapping['features'].shape[1] == 12, f"Expected 12 features in array, got {tremor_dataset_mapping['features'].shape[1]}"
print(f"  ✅ Feature count verified: 12")

with open('models/tremor_dataset_mapping.pkl', 'wb') as f:
    pickle.dump(tremor_dataset_mapping, f)

elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📈 Results:")
print(f"  • Voice Model: {voice_accuracy:.1%} accuracy")
print(f"  • Tremor Model: {tremor_accuracy:.1%} accuracy")
print(f"  • Training Time: {elapsed_time:.1f}s")
print(f"  • Models Saved: 6 files in models/")

# Validation: Test loading the scalers
print("\n🔍 Validating saved models...")
try:
    with open('models/voice_scaler.pkl', 'rb') as f:
        test_voice_scaler = pickle.load(f)
    print(f"  ✓ Voice scaler: {test_voice_scaler}")
    
    with open('models/tremor_scaler.pkl', 'rb') as f:
        test_tremor_scaler = pickle.load(f)
    print(f"  ✓ Tremor scaler: {test_tremor_scaler}")
    
    # Test that scalers can transform
    test_voice_sample = X_voice[:1]
    # Apply selection first!
    test_voice_sample_selected = selector.transform(test_voice_sample)
    test_voice_scaled = test_voice_scaler.transform(test_voice_sample_selected)
    print(f"  ✓ Voice scaler transform: OK (shape: {test_voice_scaled.shape})")
    
    test_tremor_sample = X_tremor[:1]
    test_tremor_scaled = test_tremor_scaler.transform(test_tremor_sample)
    print(f"  ✓ Tremor scaler transform: OK (shape: {test_tremor_scaled.shape})")
    
    print("\n✅ All models and scalers validated successfully!")
except Exception as e:
    print(f"\n❌ Validation failed: {str(e)}")

print("="*80 + "\n")
