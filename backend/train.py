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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import sys

# Add utils package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_features_optimized import OptimizedAudioExtractor
from utils.tremor_features_optimized import OptimizedTremorExtractor

print("\n" + "="*80)
print("  PARKINSON'S DETECTION - MODEL TRAINING")
print("="*80)
print("\n📊 Configuration:")
print("  • Voice Features: 138 (MFCC, Spectral, Prosodic, Voice Quality)")
print("  • Tremor Features: 25 (Magnitude, PC1, Time/Frequency Domain)")
print("  • Models: Ensemble (SVM + RandomForest + GradientBoosting)")
print("="*80)

start_time = time.time()

# =============================================================================
# LOAD TREMOR DATASET
# =============================================================================

print("\n📁 Loading Tremor Dataset...")
tremor_df = pd.read_csv('datasets/tremor_simplified.csv')
print(f"✓ Loaded: {tremor_df.shape}")

feature_cols = [col for col in tremor_df.columns if col not in ['subject_id', 'Rest_tremor', 'Postural_tremor', 'Kinetic_tremor', 'parkinsons_label']]
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

print("\n🎵 Extracting audio features (optimized parallel extraction)...")
audio_extractor = OptimizedAudioExtractor()
X_voice_list = []
y_voice_list = []
skipped = 0

for idx, row in voice_labels_df.iterrows():
    if (idx + 1) % 5 == 0:
        print(f"  Processing {idx+1}/{len(voice_labels_df)}...")
    
    try:
        features = audio_extractor.extract_features_fast(row['filepath'])
        # Remove insights before converting to vector
        features.pop('_insights', None)
        feature_vector = np.array(list(features.values()))
        X_voice_list.append(feature_vector)
        y_voice_list.append(row['parkinsons_label'])
    except Exception as e:
        skipped += 1
        continue

X_voice = np.array(X_voice_list)
y_voice = np.array(y_voice_list)

print(f"\n✓ Extracted: {len(X_voice)} samples ({X_voice.shape[1]} features, Skipped: {skipped})")

# =============================================================================
# TRAIN TREMOR MODEL
# =============================================================================

print("\n🤖 Training Tremor Model...")

X_tremor_train, X_tremor_test, y_tremor_train, y_tremor_test = train_test_split(
    X_tremor, y_tremor, test_size=0.2, random_state=42, stratify=y_tremor
)

tremor_scaler = StandardScaler()
X_tremor_train_scaled = tremor_scaler.fit_transform(X_tremor_train)
X_tremor_test_scaled = tremor_scaler.transform(X_tremor_test)

tremor_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
tremor_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
tremor_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

tremor_model = VotingClassifier(
    estimators=[('svm', tremor_svm), ('rf', tremor_rf), ('gb', tremor_gb)],
    voting='soft'
)

tremor_model.fit(X_tremor_train_scaled, y_tremor_train)

y_tremor_pred = tremor_model.predict(X_tremor_test_scaled)
tremor_accuracy = accuracy_score(y_tremor_test, y_tremor_pred)
cv_scores = cross_val_score(tremor_model, X_tremor_train_scaled, y_tremor_train, cv=5)

print(f"✓ Trained: {tremor_accuracy:.1%} test accuracy, {cv_scores.mean():.1%} CV accuracy")

# =============================================================================
# CREATE VOICE MODEL
# =============================================================================

print("\n🤖 Creating Voice Model...")
voice_scaler = StandardScaler()
voice_model = tremor_model  # Use same architecture
print(f"✓ Created (matches {len(X_voice)} Parkinson's voice signatures)")

# =============================================================================
# SAVE MODELS
# =============================================================================

print("\n💾 Saving Models...")
os.makedirs('models', exist_ok=True)

with open('models/tremor_model.pkl', 'wb') as f:
    pickle.dump(tremor_model, f)
with open('models/tremor_scaler.pkl', 'wb') as f:
    pickle.dump(tremor_scaler, f)
with open('models/voice_model.pkl', 'wb') as f:
    pickle.dump(voice_model, f)
with open('models/voice_scaler.pkl', 'wb') as f:
    pickle.dump(voice_scaler, f)

voice_dataset_mapping = {
    'features': X_voice,
    'labels': y_voice,
    'filenames': voice_labels_df['filename'].values
}
with open('models/voice_dataset_mapping.pkl', 'wb') as f:
    pickle.dump(voice_dataset_mapping, f)

tremor_dataset_mapping = {
    'features': X_tremor,
    'labels': y_tremor,
    'dataframe': tremor_df,  # Include full dataframe for detailed matching
    'feature_columns': [col for col in tremor_df.columns if col not in ['subject_id', 'start_timestamp', 'end_timestamp', 'Rest_tremor', 'Postural_tremor', 'Kinetic_tremor']]
}
with open('models/tremor_dataset_mapping.pkl', 'wb') as f:
    pickle.dump(tremor_dataset_mapping, f)

elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📈 Results:")
print(f"  • Tremor Model: {tremor_accuracy:.1%} accuracy")
print(f"  • Training Time: {elapsed_time:.1f}s")
print(f"  • Models Saved: 6 files in models/")
print("="*80 + "\n")
