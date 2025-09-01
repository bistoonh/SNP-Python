"""
Tests for SNP algorithm.
"""

import numpy as np
import pytest
from snp import SNP


class TestSNP:
    """Test class for SNP function."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(123)
        self.n = 50
        self.x = np.sort(np.random.uniform(0, 1, self.n))
        self.y = np.sin(2*np.pi*self.x) + np.random.normal(0, 0.1, self.n)
    
    def test_basic_functionality(self):
        """Test basic SNP functionality."""
        result = SNP(self.x, self.y, num_h_points=10, verbose=False)
        
        # Check return structure
        assert isinstance(result, dict)
        expected_keys = {'y_k_opt', 'h_start', 'k_opt', 'gcv_approx_k', 'time_elapsed'}
        assert set(result.keys()) == expected_keys
        
        # Check dimensions
        assert len(result['y_k_opt']) == self.n
        assert isinstance(result['h_start'], (int, float))
        assert isinstance(result['k_opt'], (int, np.integer))
        assert len(result['gcv_approx_k']) == 10  # k_max is 10
        assert isinstance(result['time_elapsed'], (int, float))
        
        # Check value ranges
        assert result['h_start'] > 0
        assert 1 <= result['k_opt'] <= 10
        assert result['time_elapsed'] >= 0
        assert np.all(np.isfinite(result['y_k_opt']))
        assert np.all(np.isfinite(result['gcv_approx_k']))
    
    def test_edge_cases(self):
        """Test SNP with edge cases."""
        # Test with minimal data
        x_small = np.array([1.0, 2.0, 3.0])
        y_small = np.array([1.0, 4.0, 2.0])
        
        result = SNP(x_small, y_small, num_h_points=5, verbose=False)
        assert len(result['y_k_opt']) == 3
        
        # Test input validation
        with pytest.raises(ValueError, match="same length"):
            SNP(np.array([1, 2]), np.array([1]), verbose=False)
            
        with pytest.raises(ValueError, match="non-finite"):
            SNP(np.array([1, np.nan]), np.array([1, 2]), verbose=False)
            
        with pytest.raises(ValueError, match="non-finite"):
            SNP(np.array([1, 2]), np.array([1, np.nan]), verbose=False)
            
        with pytest.raises(ValueError, match="positive"):
            SNP(self.x, self.y, num_h_points=0, verbose=False)
    
    def test_reasonable_smoothing(self):
        """Test that SNP produces reasonable smoothing."""
        # Test with known smooth function
        np.random.seed(456)
        x = np.linspace(0, 1, 30)
        true_y = np.sin(2*np.pi*x)
        y_noisy = true_y + np.random.normal(0, 0.05, 30)
        
        result = SNP(x, y_noisy, num_h_points=10, verbose=False)
        
        # Smoothed result should be closer to true function than noisy observations
        mse_original = np.mean((y_noisy - true_y) ** 2)
        mse_smoothed = np.mean((result['y_k_opt'] - true_y) ** 2)
        
        assert mse_smoothed < mse_original
    
    def test_reproducibility(self):
        """Test that SNP is reproducible with same random seed."""
        # Run twice with same seed
        np.random.seed(100)
        result1 = SNP(self.x, self.y, num_h_points=5, verbose=False)
        
        np.random.seed(100)
        result2 = SNP(self.x, self.y, num_h_points=5, verbose=False)
        
        np.testing.assert_array_equal(result1['y_k_opt'], result2['y_k_opt'])
        assert result1['h_start'] == result2['h_start']
        assert result1['k_opt'] == result2['k_opt']
    
    def test_different_data_sizes(self):
        """Test SNP with different data sizes."""
        sizes = [10, 30, 100, 200]
        
        for n in sizes:
            np.random.seed(42)
            x = np.sort(np.random.uniform(0, 1, n))
            y = x**2 + np.random.normal(0, 0.1, n)
            
            result = SNP(x, y, num_h_points=5, verbose=False)
            
            assert len(result['y_k_opt']) == n
            assert result['h_start'] > 0
            assert 1 <= result['k_opt'] <= 10
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test num_h_points validation
        with pytest.raises(ValueError):
            SNP(self.x, self.y, num_h_points=-1, verbose=False)
            
        # Test that function works with different num_h_points
        for nh in [5, 20, 100]:
            result = SNP(self.x, self.y, num_h_points=nh, verbose=False)
            assert len(result['y_k_opt']) == self.n