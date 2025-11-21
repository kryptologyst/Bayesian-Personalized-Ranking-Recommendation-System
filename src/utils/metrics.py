"""Evaluation metrics for recommendation systems."""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import logging


class RecommendationMetrics:
    """Class for computing recommendation evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def precision_at_k(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """Compute Precision@K.
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average Precision@K across all users
        """
        precisions = []
        
        for user_recs, user_gt in zip(recommendations, ground_truth):
            if len(user_recs) == 0:
                precisions.append(0.0)
                continue
                
            # Take top-k recommendations
            top_k_recs = set(user_recs[:k])
            
            # Compute precision
            if len(top_k_recs) > 0:
                precision = len(top_k_recs.intersection(user_gt)) / len(top_k_recs)
            else:
                precision = 0.0
                
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def recall_at_k(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """Compute Recall@K.
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average Recall@K across all users
        """
        recalls = []
        
        for user_recs, user_gt in zip(recommendations, ground_truth):
            if len(user_gt) == 0:
                recalls.append(0.0)
                continue
                
            # Take top-k recommendations
            top_k_recs = set(user_recs[:k])
            
            # Compute recall
            recall = len(top_k_recs.intersection(user_gt)) / len(user_gt)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def ndcg_at_k(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """Compute NDCG@K.
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average NDCG@K across all users
        """
        ndcgs = []
        
        for user_recs, user_gt in zip(recommendations, ground_truth):
            if len(user_gt) == 0:
                ndcgs.append(0.0)
                continue
                
            # Take top-k recommendations
            top_k_recs = user_recs[:k]
            
            # Compute DCG
            dcg = 0.0
            for i, item in enumerate(top_k_recs):
                if item in user_gt:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Compute IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(k, len(user_gt))):
                idcg += 1.0 / np.log2(i + 2)
            
            # Compute NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
                
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    def map_at_k(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """Compute MAP@K (Mean Average Precision).
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average MAP@K across all users
        """
        maps = []
        
        for user_recs, user_gt in zip(recommendations, ground_truth):
            if len(user_gt) == 0:
                maps.append(0.0)
                continue
                
            # Take top-k recommendations
            top_k_recs = user_recs[:k]
            
            # Compute average precision
            relevant_count = 0
            precision_sum = 0.0
            
            for i, item in enumerate(top_k_recs):
                if item in user_gt:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            if relevant_count > 0:
                avg_precision = precision_sum / min(k, len(user_gt))
            else:
                avg_precision = 0.0
                
            maps.append(avg_precision)
        
        return np.mean(maps)
    
    def hit_rate_at_k(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k: int
    ) -> float:
        """Compute Hit Rate@K.
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K across all users
        """
        hits = 0
        total_users = 0
        
        for user_recs, user_gt in zip(recommendations, ground_truth):
            if len(user_gt) == 0:
                continue
                
            total_users += 1
            
            # Take top-k recommendations
            top_k_recs = set(user_recs[:k])
            
            # Check if any relevant item is in top-k
            if len(top_k_recs.intersection(user_gt)) > 0:
                hits += 1
        
        return hits / total_users if total_users > 0 else 0.0
    
    def coverage(
        self,
        recommendations: List[List[int]],
        n_items: int
    ) -> float:
        """Compute catalog coverage.
        
        Args:
            recommendations: List of recommendation lists for each user
            n_items: Total number of items in catalog
            
        Returns:
            Coverage score (fraction of items recommended)
        """
        recommended_items = set()
        for user_recs in recommendations:
            recommended_items.update(user_recs)
        
        return len(recommended_items) / n_items
    
    def diversity(
        self,
        recommendations: List[List[int]],
        item_similarities: np.ndarray = None
    ) -> float:
        """Compute intra-list diversity.
        
        Args:
            recommendations: List of recommendation lists for each user
            item_similarities: Item similarity matrix (optional)
            
        Returns:
            Average diversity score
        """
        diversities = []
        
        for user_recs in recommendations:
            if len(user_recs) < 2:
                diversities.append(0.0)
                continue
            
            if item_similarities is not None:
                # Feature-based diversity using similarity matrix
                similarities = []
                for i in range(len(user_recs)):
                    for j in range(i + 1, len(user_recs)):
                        sim = item_similarities[user_recs[i], user_recs[j]]
                        similarities.append(sim)
                
                diversity = 1 - np.mean(similarities) if similarities else 0
            else:
                # Simple diversity based on unique items
                diversity = len(set(user_recs)) / len(user_recs)
            
            diversities.append(diversity)
        
        return np.mean(diversities)
    
    def novelty(
        self,
        recommendations: List[List[int]],
        item_popularity: np.ndarray
    ) -> float:
        """Compute average novelty of recommendations.
        
        Args:
            recommendations: List of recommendation lists for each user
            item_popularity: Popularity scores for each item
            
        Returns:
            Average novelty score (higher = more novel)
        """
        novelties = []
        
        for user_recs in recommendations:
            user_novelty = np.mean([-np.log2(item_popularity[item] + 1e-8) for item in user_recs])
            novelties.append(user_novelty)
        
        return np.mean(novelties)
    
    def evaluate_all(
        self,
        recommendations: List[List[int]],
        ground_truth: List[Set[int]],
        k_values: List[int],
        n_items: int,
        item_popularity: np.ndarray = None,
        item_similarities: np.ndarray = None
    ) -> Dict[str, Dict[int, float]]:
        """Evaluate all metrics.
        
        Args:
            recommendations: List of recommendation lists for each user
            ground_truth: List of ground truth item sets for each user
            k_values: List of k values to evaluate
            n_items: Total number of items
            item_popularity: Item popularity scores (optional)
            item_similarities: Item similarity matrix (optional)
            
        Returns:
            Dictionary of metrics results
        """
        results = {}
        
        # Ranking metrics
        for metric_name, metric_func in [
            ("precision", self.precision_at_k),
            ("recall", self.recall_at_k),
            ("ndcg", self.ndcg_at_k),
            ("map", self.map_at_k),
            ("hit_rate", self.hit_rate_at_k)
        ]:
            results[metric_name] = {}
            for k in k_values:
                results[metric_name][k] = metric_func(recommendations, ground_truth, k)
        
        # Diversity and coverage metrics
        results["coverage"] = self.coverage(recommendations, n_items)
        results["diversity"] = self.diversity(recommendations, item_similarities)
        
        if item_popularity is not None:
            results["novelty"] = self.novelty(recommendations, item_popularity)
        
        return results
    
    def print_results(self, results: Dict[str, Dict[int, float]], k_values: List[int]) -> None:
        """Print evaluation results in a formatted table.
        
        Args:
            results: Dictionary of metrics results
            k_values: List of k values
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Ranking metrics
        print("\nRanking Metrics:")
        print("-" * 50)
        for metric_name in ["precision", "recall", "ndcg", "map", "hit_rate"]:
            if metric_name in results:
                print(f"\n{metric_name.upper()}:")
                for k in k_values:
                    if k in results[metric_name]:
                        print(f"  @{k:2d}: {results[metric_name][k]:.4f}")
        
        # Other metrics
        print("\nOther Metrics:")
        print("-" * 50)
        for metric_name in ["coverage", "diversity", "novelty"]:
            if metric_name in results:
                print(f"{metric_name.upper()}: {results[metric_name]:.4f}")
        
        print("="*80)
