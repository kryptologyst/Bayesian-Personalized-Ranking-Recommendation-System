"""Data pipeline for BPR recommendation system."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import logging


class BPRDataset(Dataset):
    """Dataset for BPR training with pairwise sampling."""
    
    def __init__(
        self,
        interactions: np.ndarray,
        user_item_map: Dict[int, List[int]],
        n_negatives: int = 1,
        seed: int = 42
    ):
        """Initialize BPR dataset.
        
        Args:
            interactions: User-item interaction matrix
            user_item_map: Dictionary mapping user_id to list of positive items
            n_negatives: Number of negative samples per positive sample
            seed: Random seed for reproducibility
        """
        self.interactions = interactions
        self.user_item_map = user_item_map
        self.n_negatives = n_negatives
        self.n_users, self.n_items = interactions.shape
        
        # Set random seed
        np.random.seed(seed)
        
        # Create training samples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[int, int, int]]:
        """Create training samples (user, positive_item, negative_item)."""
        samples = []
        
        for user_id, positive_items in self.user_item_map.items():
            for pos_item in positive_items:
                # Sample negative items
                for _ in range(self.n_negatives):
                    neg_item = self._sample_negative_item(user_id, positive_items)
                    samples.append((user_id, pos_item, neg_item))
        
        return samples
    
    def _sample_negative_item(self, user_id: int, positive_items: List[int]) -> int:
        """Sample a negative item for the user."""
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in positive_items:
                return neg_item
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_id, pos_item, neg_item = self.samples[idx]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )


class DataGenerator:
    """Generate synthetic interaction data for BPR training."""
    
    def __init__(self, config):
        """Initialize data generator.
        
        Args:
            config: Configuration object containing data parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_interactions(self) -> Tuple[np.ndarray, Dict[int, List[int]], Dict[int, List[int]]]:
        """Generate synthetic user-item interactions.
        
        Returns:
            Tuple of (interaction_matrix, user_item_map, item_user_map)
        """
        n_users = self.config.data.n_users
        n_items = self.config.data.n_items
        interaction_prob = self.config.data.interaction_probability
        
        # Generate interaction matrix
        interactions = np.random.binomial(1, interaction_prob, (n_users, n_items))
        
        # Filter users and items with minimum interactions
        user_counts = np.sum(interactions, axis=1)
        item_counts = np.sum(interactions, axis=0)
        
        # Keep users and items with sufficient interactions
        valid_users = user_counts >= self.config.data.min_interactions_per_user
        valid_items = item_counts >= self.config.data.min_interactions_per_item
        
        interactions = interactions[valid_users][:, valid_items]
        
        self.logger.info(f"Generated interactions: {interactions.shape}")
        self.logger.info(f"Interaction density: {np.mean(interactions):.4f}")
        
        # Create user-item and item-user mappings
        user_item_map = {}
        item_user_map = {}
        
        for user_id in range(interactions.shape[0]):
            positive_items = np.where(interactions[user_id] > 0)[0].tolist()
            if positive_items:
                user_item_map[user_id] = positive_items
                
                for item_id in positive_items:
                    if item_id not in item_user_map:
                        item_user_map[item_id] = []
                    item_user_map[item_id].append(user_id)
        
        return interactions, user_item_map, item_user_map
    
    def split_data(
        self,
        interactions: np.ndarray,
        user_item_map: Dict[int, List[int]],
        test_ratio: float = None,
        val_ratio: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, List[int]], Dict[int, List[int]]]:
        """Split data into train/validation/test sets.
        
        Args:
            interactions: User-item interaction matrix
            user_item_map: Dictionary mapping user_id to positive items
            test_ratio: Ratio of test data
            val_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_interactions, val_interactions, test_interactions, train_user_item_map, val_user_item_map)
        """
        if test_ratio is None:
            test_ratio = self.config.data.test_ratio
        if val_ratio is None:
            val_ratio = self.config.data.val_ratio
            
        train_interactions = interactions.copy()
        val_interactions = np.zeros_like(interactions)
        test_interactions = np.zeros_like(interactions)
        
        train_user_item_map = {}
        val_user_item_map = {}
        
        for user_id, positive_items in user_item_map.items():
            if len(positive_items) < 2:
                # If user has only one interaction, keep it in training
                train_user_item_map[user_id] = positive_items
                continue
                
            # Split user's interactions
            train_items, temp_items = train_test_split(
                positive_items, 
                test_size=test_ratio + val_ratio, 
                random_state=self.config.random_seed
            )
            
            if len(temp_items) > 1:
                val_items, test_items = train_test_split(
                    temp_items,
                    test_size=test_ratio / (test_ratio + val_ratio),
                    random_state=self.config.random_seed
                )
            else:
                val_items = temp_items
                test_items = []
            
            # Update interaction matrices
            train_user_item_map[user_id] = train_items
            if val_items:
                val_user_item_map[user_id] = val_items
                for item_id in val_items:
                    val_interactions[user_id, item_id] = 1
                    train_interactions[user_id, item_id] = 0
            
            if test_items:
                for item_id in test_items:
                    test_interactions[user_id, item_id] = 1
                    train_interactions[user_id, item_id] = 0
        
        self.logger.info(f"Data split - Train: {np.sum(train_interactions)}, "
                        f"Val: {np.sum(val_interactions)}, Test: {np.sum(test_interactions)}")
        
        return train_interactions, val_interactions, test_interactions, train_user_item_map, val_user_item_map


class DataLoader:
    """Data loader for BPR training."""
    
    def __init__(self, config):
        """Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_dataloaders(
        self,
        train_interactions: np.ndarray,
        val_interactions: np.ndarray,
        train_user_item_map: Dict[int, List[int]],
        val_user_item_map: Dict[int, List[int]]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training and validation.
        
        Args:
            train_interactions: Training interaction matrix
            val_interactions: Validation interaction matrix
            train_user_item_map: Training user-item mapping
            val_user_item_map: Validation user-item mapping
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Create datasets
        train_dataset = BPRDataset(
            train_interactions,
            train_user_item_map,
            n_negatives=1,
            seed=self.config.random_seed
        )
        
        val_dataset = BPRDataset(
            val_interactions,
            val_user_item_map,
            n_negatives=1,
            seed=self.config.random_seed
        )
        
        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Created dataloaders - Train: {len(train_dataset)} samples, "
                        f"Val: {len(val_dataset)} samples")
        
        return train_dataloader, val_dataloader
