"""Utility functions for the BPR recommendation system."""

import random
import numpy as np
import torch
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(log_dir) / 'bpr.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_model(model: torch.nn.Module, path: str, metadata: Dict[str, Any] = None) -> None:
    """Save model with metadata.
    
    Args:
        model: PyTorch model to save
        path: Path to save the model
        metadata: Additional metadata to save
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        save_dict.update(metadata)
    
    torch.save(save_dict, path)


def load_model(model: torch.nn.Module, path: str) -> Dict[str, Any]:
    """Load model from checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to the checkpoint
        
    Returns:
        Dictionary containing model state and metadata
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def calculate_popularity_bias(interactions: np.ndarray) -> float:
    """Calculate popularity bias in the dataset.
    
    Args:
        interactions: User-item interaction matrix
        
    Returns:
        Popularity bias score (higher = more biased)
    """
    item_popularity = np.sum(interactions, axis=0)
    total_interactions = np.sum(interactions)
    
    # Calculate Gini coefficient for popularity distribution
    sorted_popularity = np.sort(item_popularity)
    n = len(sorted_popularity)
    cumsum = np.cumsum(sorted_popularity)
    
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    return gini


def calculate_coverage(predictions: List[List[int]], n_items: int) -> float:
    """Calculate catalog coverage.
    
    Args:
        predictions: List of recommendation lists for each user
        n_items: Total number of items in catalog
        
    Returns:
        Coverage score (fraction of items recommended)
    """
    recommended_items = set()
    for user_predictions in predictions:
        recommended_items.update(user_predictions)
    
    return len(recommended_items) / n_items


def calculate_novelty(predictions: List[List[int]], item_popularity: np.ndarray) -> float:
    """Calculate average novelty of recommendations.
    
    Args:
        predictions: List of recommendation lists for each user
        item_popularity: Popularity scores for each item
        
    Returns:
        Average novelty score (higher = more novel)
    """
    novelty_scores = []
    for user_predictions in predictions:
        user_novelty = np.mean([-np.log2(item_popularity[item] + 1e-8) for item in user_predictions])
        novelty_scores.append(user_novelty)
    
    return np.mean(novelty_scores)


def calculate_diversity(predictions: List[List[int]], item_features: np.ndarray = None) -> float:
    """Calculate intra-list diversity.
    
    Args:
        predictions: List of recommendation lists for each user
        item_features: Item feature matrix (optional)
        
    Returns:
        Average diversity score
    """
    if item_features is None:
        # Simple diversity based on unique items
        diversities = []
        for user_predictions in predictions:
            diversity = len(set(user_predictions)) / len(user_predictions) if user_predictions else 0
            diversities.append(diversity)
        return np.mean(diversities)
    
    # Feature-based diversity using cosine similarity
    diversities = []
    for user_predictions in predictions:
        if len(user_predictions) < 2:
            diversities.append(0.0)
            continue
            
        user_features = item_features[user_predictions]
        similarities = []
        for i in range(len(user_features)):
            for j in range(i + 1, len(user_features)):
                sim = np.dot(user_features[i], user_features[j]) / (
                    np.linalg.norm(user_features[i]) * np.linalg.norm(user_features[j])
                )
                similarities.append(sim)
        
        diversity = 1 - np.mean(similarities) if similarities else 0
        diversities.append(diversity)
    
    return np.mean(diversities)
