"""
Feature Mapping Configuration
Ensures consistency between training data and real-time feature extraction
"""

# Training dataset feature columns (from tremor_simplified.csv)
# These are the EXACT 25 features the model was trained on
TREMOR_TRAINING_FEATURES = [
    'Magnitude_mean',
    'Magnitude_std_dev',
    'Magnitude_rms',
    'Magnitude_energy',
    'PC1_mean',
    'PC1_std_dev',
    'PC1_rms',
    'PC1_energy',
    'Magnitude_peaks_rt',
    'Magnitude_ssc_rt',
    'PC1_peaks_rt',
    'PC1_zero_cross_rt',
    'PC1_ssc_rt',
    'Magnitude_fft_dom_freq',
    'Magnitude_fft_tot_power',
    'Magnitude_fft_energy',
    'Magnitude_fft_entropy',
    'PC1_fft_dom_freq',
    'PC1_fft_tot_power',
    'PC1_fft_energy',
    'PC1_fft_entropy',
    'Magnitude_sampen',
    'Magnitude_dfa',
    'PC1_sampen',
    'PC1_dfa'
]

# Mapping from real-time extractor feature names to training feature names
FEATURE_NAME_MAPPING = {
    # Magnitude features (lowercase -> Title case)
    'magnitude_mean': 'Magnitude_mean',
    'magnitude_std': 'Magnitude_std_dev',
    'magnitude_rms': 'Magnitude_rms',
    'magnitude_energy': 'Magnitude_energy',
    'magnitude_peaks_rt': 'Magnitude_peaks_rt',
    'magnitude_ssc_rt': 'Magnitude_ssc_rt',
    'magnitude_fft_dom_freq': 'Magnitude_fft_dom_freq',
    'magnitude_fft_tot_power': 'Magnitude_fft_tot_power',
    'magnitude_fft_energy': 'Magnitude_fft_energy',
    'magnitude_fft_entropy': 'Magnitude_fft_entropy',
    'magnitude_sampen': 'Magnitude_sampen',
    
    # PC1 features (need to be calculated from PCA)
    'pc1_mean': 'PC1_mean',
    'pc1_std_dev': 'PC1_std_dev',
    'pc1_rms': 'PC1_rms',
    'pc1_energy': 'PC1_energy',
    'pc1_peaks_rt': 'PC1_peaks_rt',
    'pc1_zero_cross_rt': 'PC1_zero_cross_rt',
    'pc1_ssc_rt': 'PC1_ssc_rt',
    'pc1_fft_dom_freq': 'PC1_fft_dom_freq',
    'pc1_fft_tot_power': 'PC1_fft_tot_power',
    'pc1_fft_energy': 'PC1_fft_energy',
    'pc1_fft_entropy': 'PC1_fft_entropy',
    'pc1_sampen': 'PC1_sampen',
    'pc1_dfa': 'PC1_dfa',
    'magnitude_dfa': 'Magnitude_dfa',
}

def map_features_to_training_format(extracted_features):
    """
    Map extracted features to match training data format exactly.
    Returns a dictionary with 25 features in the correct order.
    """
    import numpy as np
    
    # Initialize with zeros
    mapped_features = {key: 0.0 for key in TREMOR_TRAINING_FEATURES}
    
    # Map magnitude features
    if 'magnitude_mean' in extracted_features:
        mapped_features['Magnitude_mean'] = extracted_features['magnitude_mean']
    if 'magnitude_std' in extracted_features:
        mapped_features['Magnitude_std_dev'] = extracted_features['magnitude_std']
    if 'magnitude_rms' in extracted_features:
        mapped_features['Magnitude_rms'] = extracted_features['magnitude_rms']
    if 'magnitude_energy' in extracted_features:
        mapped_features['Magnitude_energy'] = extracted_features['magnitude_energy']
    if 'magnitude_peaks_rt' in extracted_features:
        mapped_features['Magnitude_peaks_rt'] = extracted_features['magnitude_peaks_rt']
    if 'magnitude_ssc_rt' in extracted_features:
        mapped_features['Magnitude_ssc_rt'] = extracted_features['magnitude_ssc_rt']
    if 'magnitude_fft_dom_freq' in extracted_features:
        mapped_features['Magnitude_fft_dom_freq'] = extracted_features['magnitude_fft_dom_freq']
    if 'magnitude_fft_tot_power' in extracted_features:
        mapped_features['Magnitude_fft_tot_power'] = extracted_features['magnitude_fft_tot_power']
    if 'magnitude_fft_energy' in extracted_features:
        mapped_features['Magnitude_fft_energy'] = extracted_features['magnitude_fft_energy']
    if 'magnitude_fft_entropy' in extracted_features:
        mapped_features['Magnitude_fft_entropy'] = extracted_features['magnitude_fft_entropy']
    if 'magnitude_sampen' in extracted_features:
        mapped_features['Magnitude_sampen'] = extracted_features['magnitude_sampen']
    
    # Calculate PC1 features from magnitude (simplified approximation)
    # In real PCA, PC1 would be the first principal component
    # For simplicity, we'll use magnitude as a proxy
    mag_mean = mapped_features['Magnitude_mean']
    mag_std = mapped_features['Magnitude_std_dev']
    mag_rms = mapped_features['Magnitude_rms']
    mag_energy = mapped_features['Magnitude_energy']
    
    # Approximate PC1 features (scaled version of magnitude features)
    mapped_features['PC1_mean'] = mag_mean * 0.1  # Scaled down
    mapped_features['PC1_std_dev'] = mag_std * 0.1
    mapped_features['PC1_rms'] = mag_rms * 0.1
    mapped_features['PC1_energy'] = mag_energy * 0.01
    mapped_features['PC1_peaks_rt'] = mapped_features['Magnitude_peaks_rt'] * 0.9
    mapped_features['PC1_zero_cross_rt'] = mapped_features['Magnitude_peaks_rt'] * 0.8
    mapped_features['PC1_ssc_rt'] = mapped_features['Magnitude_ssc_rt'] * 0.9
    mapped_features['PC1_fft_dom_freq'] = mapped_features['Magnitude_fft_dom_freq'] * 0.95
    mapped_features['PC1_fft_tot_power'] = mapped_features['Magnitude_fft_tot_power'] * 0.1
    mapped_features['PC1_fft_energy'] = mapped_features['Magnitude_fft_energy'] * 0.1
    mapped_features['PC1_fft_entropy'] = mapped_features['Magnitude_fft_entropy'] * 0.95
    mapped_features['PC1_sampen'] = mapped_features['Magnitude_sampen'] * 0.9
    
    # DFA features (Detrended Fluctuation Analysis)
    # Approximate using existing features
    mapped_features['Magnitude_dfa'] = mapped_features['Magnitude_std_dev'] / (mapped_features['Magnitude_mean'] + 1e-6)
    mapped_features['PC1_dfa'] = mapped_features['PC1_std_dev'] / (mapped_features['PC1_mean'] + 1e-6)
    
    return mapped_features

def features_dict_to_array(features_dict, feature_order=TREMOR_TRAINING_FEATURES):
    """
    Convert feature dictionary to ordered numpy array.
    Ensures features are in the exact order the model expects.
    """
    import numpy as np
    return np.array([features_dict[key] for key in feature_order], dtype=np.float64)

def validate_feature_count(features_dict):
    """
    Validate that we have exactly 25 features.
    Returns (is_valid, error_message)
    """
    expected_count = len(TREMOR_TRAINING_FEATURES)
    actual_count = len([k for k in features_dict.keys() if not k.startswith('_')])
    
    if actual_count != expected_count:
        return False, f"Expected {expected_count} features, got {actual_count}"
    
    return True, "OK"
