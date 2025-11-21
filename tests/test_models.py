"""Unit tests for BPR recommendation system."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from src.models.bpr import BPRModel, BPRLoss, BPRTrainer
from src.models.baselines import PopularityBaseline, UserKNNBaseline, ItemKNNBaseline
from src.utils.metrics import RecommendationMetrics
from src.utils.config import Config, ModelConfig, DataConfig, EvaluationConfig


class TestBPRModel:
    """Test BPR model functionality."""
    
    def test_model_initialization(self):
        """Test BPR model initialization."""
        model = BPRModel(n_users=100, n_items=50, embedding_dim=32)
        
        assert model.n_users == 100
        assert model.n_items == 50
        assert model.embedding_dim == 32
        assert model.user_embedding.num_embeddings == 100
        assert model.item_embedding.num_embeddings == 50
    
    def test_forward_pass(self):
        """Test BPR model forward pass."""
        model = BPRModel(n_users=10, n_items=20, embedding_dim=16)
        
        user_ids = torch.tensor([0, 1, 2])
        pos_item_ids = torch.tensor([5, 10, 15])
        neg_item_ids = torch.tensor([8, 12, 18])
        
        pos_scores, neg_scores, reg_loss = model(user_ids, pos_item_ids, neg_item_ids)
        
        assert pos_scores.shape == (3,)
        assert neg_scores.shape == (3,)
        assert reg_loss.item() >= 0
    
    def test_predict(self):
        """Test BPR model prediction."""
        model = BPRModel(n_users=10, n_items=20, embedding_dim=16)
        
        user_ids = torch.tensor([0, 1])
        item_ids = torch.tensor([5, 10])
        
        scores = model.predict(user_ids, item_ids)
        
        assert scores.shape == (2,)
        assert torch.all(torch.isfinite(scores))


class TestBPRLoss:
    """Test BPR loss function."""
    
    def test_bpr_loss(self):
        """Test BPR loss computation."""
        loss_fn = BPRLoss()
        
        pos_scores = torch.tensor([2.0, 1.0])
        neg_scores = torch.tensor([0.5, 0.0])
        reg_loss = torch.tensor(0.1)
        
        loss = loss_fn(pos_scores, neg_scores, reg_loss)
        
        assert loss.item() > 0
        assert torch.isfinite(loss)


class TestBaselineModels:
    """Test baseline recommendation models."""
    
    def test_popularity_baseline(self):
        """Test popularity baseline."""
        interactions = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1]
        ])
        
        baseline = PopularityBaseline()
        baseline.fit(interactions)
        
        recommendations, scores = baseline.recommend(0, k=2)
        
        assert len(recommendations) <= 2
        assert len(scores) == len(recommendations)
    
    def test_user_knn_baseline(self):
        """Test user-KNN baseline."""
        interactions = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1]
        ])
        
        baseline = UserKNNBaseline(k=2)
        baseline.fit(interactions)
        
        recommendations, scores = baseline.recommend(0, k=2)
        
        assert len(recommendations) <= 2
        assert len(scores) == len(recommendations)
    
    def test_item_knn_baseline(self):
        """Test item-KNN baseline."""
        interactions = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1]
        ])
        
        baseline = ItemKNNBaseline(k=2)
        baseline.fit(interactions)
        
        recommendations, scores = baseline.recommend(0, k=2)
        
        assert len(recommendations) <= 2
        assert len(scores) == len(recommendations)


class TestRecommendationMetrics:
    """Test recommendation evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommendations = [[0, 1, 2], [1, 2, 3]]
        ground_truth = [{0, 1}, {1, 3}]
        
        metrics = RecommendationMetrics()
        precision = metrics.precision_at_k(recommendations, ground_truth, k=2)
        
        assert 0 <= precision <= 1
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommendations = [[0, 1, 2], [1, 2, 3]]
        ground_truth = [{0, 1}, {1, 3}]
        
        metrics = RecommendationMetrics()
        recall = metrics.recall_at_k(recommendations, ground_truth, k=2)
        
        assert 0 <= recall <= 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommendations = [[0, 1, 2], [1, 2, 3]]
        ground_truth = [{0, 1}, {1, 3}]
        
        metrics = RecommendationMetrics()
        ndcg = metrics.ndcg_at_k(recommendations, ground_truth, k=2)
        
        assert 0 <= ndcg <= 1
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommendations = [[0, 1], [1, 2], [2, 3]]
        n_items = 10
        
        metrics = RecommendationMetrics()
        coverage = metrics.coverage(recommendations, n_items)
        
        assert 0 <= coverage <= 1


class TestConfig:
    """Test configuration management."""
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig()
        
        assert config.embedding_dim == 64
        assert config.learning_rate == 0.01
        assert config.regularization == 0.01
    
    def test_data_config(self):
        """Test data configuration."""
        config = DataConfig()
        
        assert config.n_users == 1000
        assert config.n_items == 500
        assert config.interaction_probability == 0.1
    
    def test_evaluation_config(self):
        """Test evaluation configuration."""
        config = EvaluationConfig()
        
        assert config.k_values == [5, 10, 20, 50]
        assert "precision" in config.metrics
        assert "recall" in config.metrics
    
    def test_main_config(self):
        """Test main configuration."""
        config = Config(
            model=ModelConfig(),
            data=DataConfig(),
            evaluation=EvaluationConfig()
        )
        
        assert config.random_seed == 42
        assert config.device == "cpu"
        assert config.data_dir == "data"


if __name__ == "__main__":
    pytest.main([__file__])
