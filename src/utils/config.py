"""Configuration management for BPR recommendation system."""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for BPR model."""
    embedding_dim: int = 64
    learning_rate: float = 0.01
    regularization: float = 0.01
    batch_size: int = 1024
    num_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class DataConfig:
    """Configuration for data processing."""
    n_users: int = 1000
    n_items: int = 500
    interaction_probability: float = 0.1
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    k_values: list[int] = None
    metrics: list[str] = None
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [5, 10, 20, 50]
        if self.metrics is None:
            self.metrics = ["precision", "recall", "ndcg", "map", "hit_rate"]


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    evaluation: EvaluationConfig
    random_seed: int = 42
    device: str = "cpu"
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            random_seed=config_dict.get("random_seed", 42),
            device=config_dict.get("device", "cpu"),
            data_dir=config_dict.get("data_dir", "data"),
            model_dir=config_dict.get("model_dir", "models"),
            log_dir=config_dict.get("log_dir", "logs"),
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": {
                "embedding_dim": self.model.embedding_dim,
                "learning_rate": self.model.learning_rate,
                "regularization": self.model.regularization,
                "batch_size": self.model.batch_size,
                "num_epochs": self.model.num_epochs,
                "early_stopping_patience": self.model.early_stopping_patience,
            },
            "data": {
                "n_users": self.data.n_users,
                "n_items": self.data.n_items,
                "interaction_probability": self.data.interaction_probability,
                "test_ratio": self.data.test_ratio,
                "val_ratio": self.data.val_ratio,
                "min_interactions_per_user": self.data.min_interactions_per_user,
                "min_interactions_per_item": self.data.min_interactions_per_item,
            },
            "evaluation": {
                "k_values": self.evaluation.k_values,
                "metrics": self.evaluation.metrics,
            },
            "random_seed": self.random_seed,
            "device": self.device,
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
