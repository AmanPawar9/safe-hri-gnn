#!/usr/bin/env python3
"""
Human3.6M Dataset Processing Script

This script downloads, processes, and converts the Human3.6M dataset 
for training the ST-GNN model. The processed data will be saved in HDF5 format
for efficient loading during training.

Usage:
    python process_h36m.py --data_dir /path/to/h36m --output_dir /path/to/processed
"""

import argparse
import os
import h5py
import numpy as np
import urllib.request
import zipfile
import json
from tqdm import tqdm
from typing import Tuple, List, Dict


class H36MProcessor:
    """Processor for Human3.6M dataset."""
    
    # H36M joint mapping (32 joints) - we'll extend to 34 for consistency
    H36M_JOINT_NAMES = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Neck', 'Head', 'Site', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist', 'Site2', 'Site3', 'Site4',
        'Site5', 'Site6', 'Site7', 'Site8', 'Site9', 'Site10',
        'Site11', 'Site12', 'Site13', 'Site14', 'Site15', 'Site16'
    ]
    
    # Actions to include in training
    ACTIONS = [
        'Walking', 'Eating', 'Smoking', 'Discussion', 'Directions',
        'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
        'SittingDown', 'TakingPhoto', 'Waiting', 'WalkingDog', 'WalkTogether'
    ]
    
    # Subjects for train/test split
    TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
    TEST_SUBJECTS = [9, 11]
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing H36M data
            output_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.input_seq_len = 30  # 1 second at 30 FPS
        self.output_seq_len = 45  # 1.5 seconds at 30 FPS
        
        os.makedirs(output_dir, exist_ok=True)
        
    def download_sample_data(self) -> None:
        """
        Download sample H36M data for demonstration.
        Note: Full H36M requires academic license.
        """
        print("Creating sample data for demonstration...")
        
        # Create sample data structure
        sample_dir = os.path.join(self.data_dir, 'sample')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Generate synthetic pose data that mimics H36M structure
        for subject in [1, 5]:  # Just a couple subjects for demo
            for action in ['Walking', 'Sitting']:
                for trial in [1, 2]:
                    # Generate synthetic 3D poses (1000 frames, 32 joints, 3 coords)
                    poses = self._generate_synthetic_poses(1000, 32)
                    
                    filename = f"S{subject}_{action}_{trial}.npy"
                    filepath = os.path.join(sample_dir, filename)
                    np.save(filepath, poses)
                    
        print(f"Sample data created in {sample_dir}")
    
    def _generate_synthetic_poses(self, num_frames: int, num_joints: int) -> np.ndarray:
        """Generate synthetic pose data for demonstration."""
        # Create a simple walking motion pattern
        poses = np.zeros((num_frames, num_joints, 3))
        
        for frame in range(num_frames):
            t = frame / 30.0  # Time in seconds
            
            for joint in range(num_joints):
                # Simple sinusoidal motion with different frequencies per joint
                poses[frame, joint, 0] = 0.1 * np.sin(2 * np.pi * 0.5 * t + joint * 0.1)  # X
                poses[frame, joint, 1] = 0.05 * np.cos(2 * np.pi * 0.3 * t + joint * 0.15)  # Y
                poses[frame, joint, 2] = 1.0 + 0.02 * np.sin(2 * np.pi * 1.0 * t + joint * 0.2)  # Z
        
        # Add some noise
        poses += np.random.normal(0, 0.01, poses.shape)
        
        return poses
    
    def load_pose_data(self, filepath: str) -> np.ndarray:
        """
        Load pose data from file.
        
        Args:
            filepath: Path to pose data file
            
        Returns:
            Pose data array of shape (frames, joints, 3)
        """
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                data = f['poses'][:]
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        return data
    
    def normalize_poses(self, poses: np.ndarray) -> np.ndarray:
        """
        Normalize poses by centering at hip joint.
        
        Args:
            poses: Pose array of shape (frames, joints, 3)
            
        Returns:
            Normalized pose array
        """
        # Center around hip joint (joint 0)
        hip_position = poses[:, 0:1, :]  # Shape: (frames, 1, 3)
        normalized_poses = poses - hip_position
        
        return normalized_poses
    
    def create_sequences(self, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input/output sequences for training.
        
        Args:
            poses: Pose array of shape (frames, joints, 3)
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        total_frames = poses.shape[0]
        sequence_length = self.input_seq_len + self.output_seq_len
        
        if total_frames < sequence_length:
            return np.array([]), np.array([])
        
        num_sequences = total_frames - sequence_length + 1
        
        input_sequences = []
        target_sequences = []
        
        for i in range(num_sequences):
            input_seq = poses[i:i+self.input_seq_len]  # 30 frames
            target_seq = poses[i+self.input_seq_len:i+sequence_length]  # Next 45 frames
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
        
        return np.array(input_sequences), np.array(target_sequences)
    
    def extend_to_34_joints(self, poses: np.ndarray) -> np.ndarray:
        """
        Extend H36M 32-joint format to 34 joints for consistency.
        
        Args:
            poses: Pose array with 32 joints
            
        Returns:
            Pose array with 34 joints (padded with zeros)
        """
        frames, joints, coords = poses.shape
        if joints >= 34:
            return poses[:, :34, :]
        
        # Pad with two additional joints (set to zero or interpolate)
        extended_poses = np.zeros((frames, 34, coords))
        extended_poses[:, :joints, :] = poses
        
        # Simple interpolation for the missing joints
        if joints == 32:
            # Add two interpolated joints
            extended_poses[:, 32, :] = (poses[:, 8, :] + poses[:, 9, :]) / 2  # Neck-Head midpoint
            extended_poses[:, 33, :] = (poses[:, 11, :] + poses[:, 14, :]) / 2  # Shoulder midpoint
        
        return extended_poses
    
    def process_dataset(self) -> None:
        """Process the complete dataset."""
        print("Processing Human3.6M dataset...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found. Creating sample data...")
            os.makedirs(self.data_dir, exist_ok=True)
            self.download_sample_data()
        
        all_input_sequences = []
        all_target_sequences = []
        metadata = {
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            'num_joints': 34,
            'joint_names': self.H36M_JOINT_NAMES + ['Extended1', 'Extended2'],
            'actions': [],
            'subjects': [],
            'total_sequences': 0
        }
        
        # Process each file
        data_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.npy'):
                    data_files.append(os.path.join(root, file))
        
        print(f"Found {len(data_files)} data files")
        
        for filepath in tqdm(data_files, desc="Processing files"):
            try:
                # Load poses
                poses = self.load_pose_data(filepath)
                
                # Normalize poses
                poses = self.normalize_poses(poses)
                
                # Extend to 34 joints
                poses = self.extend_to_34_joints(poses)
                
                # Create sequences
                input_seqs, target_seqs = self.create_sequences(poses)
                
                if len(input_seqs) > 0:
                    all_input_sequences.append(input_seqs)
                    all_target_sequences.append(target_seqs)
                    
                    # Update metadata
                    filename = os.path.basename(filepath)
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        subject = parts[0]
                        action = parts[1]
                        if subject not in metadata['subjects']:
                            metadata['subjects'].append(subject)
                        if action not in metadata['actions']:
                            metadata['actions'].append(action)
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # Concatenate all sequences
        if all_input_sequences:
            input_data = np.concatenate(all_input_sequences, axis=0)
            target_data = np.concatenate(all_target_sequences, axis=0)
            
            metadata['total_sequences'] = input_data.shape[0]
            
            print(f"Total sequences created: {metadata['total_sequences']}")
            print(f"Input shape: {input_data.shape}")
            print(f"Target shape: {target_data.shape}")
            
            # Save processed data
            output_file = os.path.join(self.output_dir, 'h36m_processed.h5')
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('input_sequences', data=input_data)
                f.create_dataset('target_sequences', data=target_data)
                f.create_dataset('metadata', data=json.dumps(metadata))
            
            print(f"Processed data saved to {output_file}")
        else:
            print("No valid sequences were created!")
    
    def create_adjacency_matrix(self) -> np.ndarray:
        """Create skeletal adjacency matrix for H36M."""
        # Define skeletal connections for H36M joints
        connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9),  # Spine
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16),  # Right arm
        ]
        
        adj_matrix = np.zeros((34, 34))
        
        # Add connections
        for i, j in connections:
            if i < 34 and j < 34:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Add self-connections
        np.fill_diagonal(adj_matrix, 1)
        
        return adj_matrix


def main():
    parser = argparse.ArgumentParser(description="Process Human3.6M dataset for ST-GNN training")
    parser.add_argument('--data_dir', type=str, default='./data/h36m', 
                        help="Directory containing H36M raw data")
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help="Directory to save processed data")
    parser.add_argument('--create_sample', action='store_true',
                        help="Create sample synthetic data for demonstration")
    
    args = parser.parse_args()
    
    # Create processor
    processor = H36MProcessor(args.data_dir, args.output_dir)
    
    if args.create_sample:
        processor.download_sample_data()
    
    # Process dataset
    processor.process_dataset()
    
    # Save adjacency matrix
    adj_matrix = processor.create_adjacency_matrix()
    adj_file = os.path.join(args.output_dir, 'adjacency_matrix.npy')
    np.save(adj_file, adj_matrix)
    print(f"Adjacency matrix saved to {adj_file}")
    
    print("Processing complete!")


if __name__ == '__main__':
    main()
