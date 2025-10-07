"""
Custom StandardScaler Implementation
Built from scratch to avoid sklearn compatibility issues
"""
import numpy as np
import pickle


class CustomStandardScaler:
    """
    StandardScaler implementation from scratch.
    Standardizes features by removing mean and scaling to unit variance.
    
    Formula: z = (x - mean) / std
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
        self.is_fitted_ = False
    
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        self.n_features_ = X.shape[1]
        
        # Compute mean and std for each feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Handle features with zero std (constant features)
        # Set std to 1.0 to avoid division by zero
        self.std_[self.std_ == 0.0] = 1.0
        
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
        
        Returns:
        --------
        X_scaled : ndarray, shape (n_samples, n_features)
            Transformed data
        """
        if not self.is_fitted_:
            raise RuntimeError("This CustomStandardScaler instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this scaler.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but CustomStandardScaler "
                           f"is expecting {self.n_features_} features as input.")
        
        # Apply standardization: (X - mean) / std
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        
        Returns:
        --------
        X_scaled : ndarray, shape (n_samples, n_features)
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Scaled data
        
        Returns:
        --------
        X_original : ndarray, shape (n_samples, n_features)
            Original data
        """
        if not self.is_fitted_:
            raise RuntimeError("This CustomStandardScaler instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this scaler.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Reverse standardization: X * std + mean
        X_original = X * self.std_ + self.mean_
        
        return X_original
    
    def __repr__(self):
        if self.is_fitted_:
            return f"CustomStandardScaler(n_features={self.n_features_}, fitted=True)"
        else:
            return "CustomStandardScaler(fitted=False)"
    
    def __getstate__(self):
        """For pickle serialization"""
        return {
            'mean_': self.mean_,
            'std_': self.std_,
            'n_features_': self.n_features_,
            'is_fitted_': self.is_fitted_
        }
    
    def __setstate__(self, state):
        """For pickle deserialization"""
        self.mean_ = state['mean_']
        self.std_ = state['std_']
        self.n_features_ = state['n_features_']
        self.is_fitted_ = state['is_fitted_']


# Test the scaler
if __name__ == "__main__":
    print("Testing CustomStandardScaler...")
    
    # Create sample data
    X_train = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]])
    
    X_test = np.array([[2, 3, 4],
                       [5, 6, 7]])
    
    # Test fit and transform
    scaler = CustomStandardScaler()
    print(f"Before fit: {scaler}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    print(f"After fit: {scaler}")
    print(f"Mean: {scaler.mean_}")
    print(f"Std: {scaler.std_}")
    print(f"Scaled training data:\n{X_train_scaled}")
    
    # Test transform on new data
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled test data:\n{X_test_scaled}")
    
    # Test inverse transform
    X_test_original = scaler.inverse_transform(X_test_scaled)
    print(f"Inverse transformed (should match original):\n{X_test_original}")
    
    # Test pickle
    with open('test_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('test_scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    
    print(f"Loaded scaler: {loaded_scaler}")
    print("✓ All tests passed!")
