"""
Dataset Matching Module
Compares incoming samples with known dataset samples to identify matches
"""

import numpy as np
import pickle
import os
import sys
from scipy.spatial.distance import cosine, euclidean
import logging

# Add parent directory to path for feature_mapper import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_mapper import map_features_to_training_format, features_dict_to_array

logger = logging.getLogger(__name__)


class DatasetMatcher:
    """Match incoming samples against known dataset samples"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.voice_mapping = None
        self.tremor_mapping = None
        
        # Load mappings if available
        self._load_mappings()
    
    def _load_mappings(self):
        """Load dataset mappings"""
        try:
            voice_mapping_path = os.path.join(self.models_dir, 'voice_dataset_mapping.pkl')
            if os.path.exists(voice_mapping_path):
                with open(voice_mapping_path, 'rb') as f:
                    self.voice_mapping = pickle.load(f)
                logger.info("✓ Loaded voice dataset mapping")
            
            tremor_mapping_path = os.path.join(self.models_dir, 'tremor_dataset_mapping.pkl')
            if os.path.exists(tremor_mapping_path):
                with open(tremor_mapping_path, 'rb') as f:
                    self.tremor_mapping = pickle.load(f)
                logger.info("✓ Loaded tremor dataset mapping")
                
        except Exception as e:
            logger.warning(f"Could not load dataset mappings: {str(e)}")
    
    def find_voice_match(self, features_vector, threshold=0.95):
        """
        Find if voice sample matches any dataset sample
        
        Args:
            features_vector: Feature vector to match
            threshold: Similarity threshold (0-1, higher = more strict)
        
        Returns:
            dict with match info or None
        """
        if self.voice_mapping is None:
            return None
        
        try:
            dataset_features = self.voice_mapping['features']
            filenames = self.voice_mapping['filenames']
            
            # Normalize vectors for comparison
            features_norm = features_vector / (np.linalg.norm(features_vector) + 1e-10)
            
            best_similarity = 0
            best_match_idx = -1
            
            # Compare with each dataset sample
            for idx, dataset_feature in enumerate(dataset_features):
                dataset_norm = dataset_feature / (np.linalg.norm(dataset_feature) + 1e-10)
                
                # Calculate cosine similarity
                similarity = 1 - cosine(features_norm, dataset_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
            
            # Check if match is good enough
            if best_similarity >= threshold:
                # Get category from labels if available
                labels = self.voice_mapping.get('labels', [])
                if best_match_idx < len(labels):
                    category = 'Parkinsons' if labels[best_match_idx] == 1 else 'Healthy'
                else:
                    category = 'Unknown'
                
                filename = filenames[best_match_idx]
                
                return {
                    'matched': True,
                    'category': category,  # 'Healthy' or 'Parkinsons'
                    'filename': filename,
                    'similarity': float(best_similarity),
                    'confidence': float(best_similarity * 100)
                }
            
            return {
                'matched': False,
                'best_similarity': float(best_similarity),
                'message': 'No close match found in dataset'
            }
            
        except Exception as e:
            logger.error(f"Error matching voice sample: {str(e)}")
            return None
    
    def find_tremor_match(self, features_vector, threshold=0.90):
        """
        Find if tremor sample matches any dataset sample
        
        Args:
            features_vector: Feature vector to match (must be 12 features)
            threshold: Similarity threshold (0-1, higher = more strict)
        
        Returns:
            dict with match info or None
        """
        if self.tremor_mapping is None:
            return None
        
        try:
            # Validate feature vector size
            if len(features_vector) != 12:
                logger.error(f"Feature vector has {len(features_vector)} features, expected 12")
                return {
                    'matched': False,
                    'best_similarity': 0.0,
                    'message': f'Invalid feature count: {len(features_vector)} (expected 12)'
                }
            
            # Check if dataframe key exists, if not use features array format
            if 'dataframe' in self.tremor_mapping:
                df = self.tremor_mapping['dataframe']
                feature_cols = self.tremor_mapping['feature_columns']
                dataset_features = df[feature_cols].values
                use_dataframe = True
            elif 'features' in self.tremor_mapping:
                # Fallback to features array format (like voice mapping)
                dataset_features = self.tremor_mapping['features']
                use_dataframe = False
            else:
                logger.error("Tremor mapping has invalid format (no 'dataframe' or 'features' key)")
                return None
            
            # Ensure dataset features also have 12 features
            if dataset_features.shape[1] != 12:
                logger.error(f"Dataset features have {dataset_features.shape[1]} features, expected 12")
                return {
                    'matched': False,
                    'best_similarity': 0.0,
                    'message': f'Dataset feature mismatch: {dataset_features.shape[1]} (expected 12)'
                }
            
            # Normalize vectors for comparison
            features_norm = features_vector / (np.linalg.norm(features_vector) + 1e-10)
            
            best_similarity = 0
            best_match_idx = -1
            
            # Compare with each dataset sample
            for idx, dataset_feature in enumerate(dataset_features):
                dataset_norm = dataset_feature / (np.linalg.norm(dataset_feature) + 1e-10)
                
                # Calculate cosine similarity
                similarity = 1 - cosine(features_norm, dataset_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
            
            # Check if match is good enough
            if best_similarity >= threshold:
                if use_dataframe:
                    # Using dataframe format with full metadata
                    matched_row = df.iloc[best_match_idx]
                    
                    # Determine tremor status
                    has_tremor = False
                    tremor_types = []
                    
                    if 'Rest_tremor' in df.columns and matched_row['Rest_tremor'] > 0:
                        has_tremor = True
                        tremor_types.append('Rest')
                    if 'Postural_tremor' in df.columns and matched_row['Postural_tremor'] > 0:
                        has_tremor = True
                        tremor_types.append('Postural')
                    if 'Kinetic_tremor' in df.columns and matched_row['Kinetic_tremor'] > 0:
                        has_tremor = True
                        tremor_types.append('Kinetic')
                    
                    category = 'Parkinsons' if has_tremor else 'Healthy'
                    
                    return {
                        'matched': True,
                        'category': category,
                        'subject_id': matched_row.get('subject_id', 'Unknown'),
                        'tremor_types': tremor_types if has_tremor else [],
                        'similarity': float(best_similarity),
                        'confidence': float(best_similarity * 100)
                    }
                else:
                    # Using simple features array format
                    labels = self.tremor_mapping.get('labels', [])
                    subject_ids = self.tremor_mapping.get('subject_ids', [])
                    
                    category = 'Parkinsons' if (labels and best_match_idx < len(labels) and labels[best_match_idx] == 1) else 'Healthy'
                    subject_id = subject_ids[best_match_idx] if (subject_ids and best_match_idx < len(subject_ids)) else 'Unknown'
                    
                    return {
                        'matched': True,
                        'category': category,
                        'subject_id': subject_id,
                        'tremor_types': [],
                        'similarity': float(best_similarity),
                        'confidence': float(best_similarity * 100)
                    }
            
            return {
                'matched': False,
                'best_similarity': float(best_similarity),
                'message': 'No close match found in dataset'
            }
            
        except Exception as e:
            logger.error(f"Error matching tremor sample: {str(e)}")
            return None
    
    def match_combined(self, voice_features=None, tremor_features=None):
        """
        Match both voice and tremor if available
        
        Returns:
            dict with combined match results
        """
        result = {
            'voice_match': None,
            'tremor_match': None,
            'overall_match': False
        }
        
        if voice_features is not None:
            result['voice_match'] = self.find_voice_match(voice_features)
        
        if tremor_features is not None:
            result['tremor_match'] = self.find_tremor_match(tremor_features)
        
        # Determine overall match
        voice_matched = result['voice_match'] and result['voice_match'].get('matched', False)
        tremor_matched = result['tremor_match'] and result['tremor_match'].get('matched', False)
        
        if voice_matched or tremor_matched:
            result['overall_match'] = True
            
            # Determine category consensus
            categories = []
            if voice_matched:
                categories.append(result['voice_match']['category'])
            if tremor_matched:
                categories.append(result['tremor_match']['category'])
            
            # Majority vote
            if categories:
                result['consensus_category'] = max(set(categories), key=categories.count)
        
        return result
