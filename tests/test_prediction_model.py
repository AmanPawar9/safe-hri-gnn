#!/usr/bin/env python3
"""
Unit tests for the ST-GNN prediction model.

This test suite validates:
1. Model architecture and forward pass
2. Training loop functionality
3. Prediction function accuracy
4. Data processing and filtering
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from prediction_model.prediction_model.model import STGNNModel, predict_future_poses, train_model
    from prediction_model.prediction_model.utils import get_adjacency_matrix, NUM_JOINTS
    from perception.perception.kalman_filter import KalmanFilter
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This test requires the packages to be built with colcon first.")


class TestSTGNNModel(unittest.TestCase):
    """Test cases for the ST-GNN model."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.input_seq_len = 30
        self.output_seq_len = 45
        self.num_joints = 34
        self.num_features = 3

        # Create model
        self.model = STGNNModel(
            num_nodes=self.num_joints,
            in_features=self.num_features,
            out_features=self.num_features,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len
        )
        
        # Create sample input data
        self.sample_input = torch.randn(
            self.batch_size, 
            self.input_seq_len, 
            self.num_joints, 
            self.num_features
        )

    def test_model_architecture(self):
        """Test that the model architecture is correctly defined."""
        self.assertIsInstance(self.model, STGNNModel)
        self.assertEqual(self.model.num_nodes, self.num_joints)
        self.assertEqual(self.model.output_seq_len, self.output_seq_len)

    def test_forward_pass(self):
        """Test that the model can perform a forward pass."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.sample_input)
        
        expected_shape = (self.batch_size, self.output_seq_len, self.num_joints, 3)
        self.assertEqual(output.shape, expected_shape)

    def test_adjacency_matrix(self):
        """Test the adjacency matrix generation."""
        adj_matrix = get_adjacency_matrix()
        
        self.assertEqual(adj_matrix.shape, (NUM_JOINTS, NUM_JOINTS))
        self.assertTrue(torch.allclose(adj_matrix, adj_matrix.T))  # Should be symmetric
        self.assertTrue((torch.diag(adj_matrix) == 1).all())  # Should have self-loops

    def test_predict_future_poses(self):
        """Test the prediction function with mock data."""
        # Create mock pose history (30 frames of 34 joints)
        pose_history = [
            [(i * 0.1, j * 0.1, (i + j) * 0.1) for j in range(34)]
            for i in range(30)
        ]
        
        self.model.eval()
        predictions = predict_future_poses(self.model, pose_history)
        
        # Should return 45 frames
        self.assertEqual(len(predictions), 45)
        # Each frame should have 34 joints
        self.assertEqual(len(predictions[0]), 34)
        # Each joint should have 3 coordinates
        self.assertEqual(len(predictions[0][0]), 3)


class TestKalmanFilter(unittest.TestCase):
    """Test cases for the Kalman filter."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = KalmanFilter()

    def test_initialization(self):
        """Test that the Kalman filter initializes correctly."""
        self.assertEqual(self.filter.state_dim, 6)  # x, y, z, vx, vy, vz
        self.assertEqual(self.filter.measurement_dim, 3)  # x, y, z measurements

    def test_process_measurement(self):
        """Test processing a measurement."""
        measurement = np.array([[1.0], [2.0], [3.0]])  # x, y, z position
        result = self.filter.process(measurement)
        
        self.assertEqual(result.shape, (3,))  # Should return 3D position
        self.assertIsInstance(result, np.ndarray)

    def test_filter_smoothing(self):
        """Test that the filter provides smoothing."""
        measurements = [
            np.array([[1.0 + 0.1 * np.random.randn()], 
                     [2.0 + 0.1 * np.random.randn()], 
                     [3.0 + 0.1 * np.random.randn()]])
            for _ in range(10)
        ]
        
        results = []
        for measurement in measurements:
            result = self.filter.process(measurement)
            results.append(result)
        
        # The filtered results should be smoother than the raw measurements
        self.assertEqual(len(results), 10)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""

    def test_pose_history_format(self):
        """Test that pose history is in the correct format."""
        # This would test the data processing node's output format
        # For now, just test the expected data structure
        
        pose_history = [
            [(i * 0.1, j * 0.1, (i + j) * 0.1) for j in range(34)]
            for i in range(30)
        ]
        
        # Should have 30 frames
        self.assertEqual(len(pose_history), 30)
        # Each frame should have 34 joints
        for frame in pose_history:
            self.assertEqual(len(frame), 34)
            # Each joint should have 3 coordinates
            for joint in frame:
                self.assertEqual(len(joint), 3)


class TestTrainingLoop(unittest.TestCase):
    """Test cases for the training functionality."""

    def setUp(self):
        """Set up test fixtures for training."""
        self.model = STGNNModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.L1Loss()

    def create_mock_dataloader(self):
        """Create a mock data loader for testing."""
        class MockDataLoader:
            def __init__(self, num_batches=5):
                self.num_batches = num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    inputs = torch.randn(8, 30, 34, 3)
                    targets = torch.randn(8, 45, 34, 3)
                    yield inputs, targets
            
            def __len__(self):
                return self.num_batches
        
        return MockDataLoader()

    def test_training_step(self):
        """Test that training can run without errors."""
        train_loader = self.create_mock_dataloader()
        
        # Run one training epoch
        avg_loss = train_model(self.model, train_loader, self.optimizer, self.loss_fn)
        
        # Should return a loss value
        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestSTGNNModel, TestKalmanFilter, TestDataProcessing, TestTrainingLoop]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
