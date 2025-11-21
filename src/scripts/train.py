"""Main training script for BPR recommendation system."""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.utils import set_seed, setup_logging, create_directories, get_device
from src.data.dataset import DataGenerator, DataLoader
from src.models.bpr import BPRModel, BPRTrainer
from src.models.baselines import (
    PopularityBaseline, UserKNNBaseline, ItemKNNBaseline, RandomBaseline
)
from src.utils.metrics import RecommendationMetrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BPR recommendation system")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Setup logging
    logger = setup_logging(config.log_dir, "INFO" if args.verbose else "WARNING")
    logger.info("Starting BPR training")
    
    # Create directories
    create_directories([config.data_dir, config.model_dir, config.log_dir])
    
    # Get device
    device = get_device(args.device)
    config.device = str(device)
    logger.info(f"Using device: {device}")
    
    # Generate data
    logger.info("Generating synthetic data")
    data_generator = DataGenerator(config)
    interactions, user_item_map, item_user_map = data_generator.generate_interactions()
    
    # Split data
    train_interactions, val_interactions, test_interactions, train_user_item_map, val_user_item_map = data_generator.split_data(
        interactions, user_item_map
    )
    
    # Create data loaders
    data_loader = DataLoader(config)
    train_dataloader, val_dataloader = data_loader.create_dataloaders(
        train_interactions, val_interactions, train_user_item_map, val_user_item_map
    )
    
    # Initialize models
    logger.info("Initializing models")
    n_users, n_items = train_interactions.shape
    
    # BPR Model
    bpr_model = BPRModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.model.embedding_dim,
        regularization=config.model.regularization
    )
    
    bpr_trainer = BPRTrainer(
        model=bpr_model,
        learning_rate=config.model.learning_rate,
        device=config.device
    )
    
    # Baseline models
    baselines = {
        "popularity": PopularityBaseline(),
        "user_knn": UserKNNBaseline(k=50),
        "item_knn": ItemKNNBaseline(k=50),
        "random": RandomBaseline(seed=config.random_seed)
    }
    
    # Train baselines
    logger.info("Training baseline models")
    for name, baseline in baselines.items():
        logger.info(f"Training {name} baseline")
        baseline.fit(train_interactions)
    
    # Train BPR model
    logger.info("Training BPR model")
    bpr_trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=config.model.num_epochs,
        early_stopping_patience=config.model.early_stopping_patience,
        verbose=args.verbose
    )
    
    # Evaluate models
    logger.info("Evaluating models")
    metrics_calculator = RecommendationMetrics()
    
    # Prepare test data
    test_user_item_map = {}
    for user_id in range(test_interactions.shape[0]):
        test_items = np.where(test_interactions[user_id] > 0)[0].tolist()
        if test_items:
            test_user_item_map[user_id] = test_items
    
    # Get recommendations for all models
    all_recommendations = {}
    k = max(config.evaluation.k_values)
    
    # BPR recommendations
    logger.info("Generating BPR recommendations")
    bpr_recommendations = []
    for user_id in range(n_users):
        if user_id in test_user_item_map:
            seen_items = set(train_user_item_map.get(user_id, []))
            item_ids, scores = bpr_trainer.get_recommendations(
                user_id, k=k, exclude_seen=True, seen_items=seen_items
            )
            bpr_recommendations.append(item_ids.tolist())
        else:
            bpr_recommendations.append([])
    
    all_recommendations["BPR"] = bpr_recommendations
    
    # Baseline recommendations
    for name, baseline in baselines.items():
        logger.info(f"Generating {name} recommendations")
        baseline_recommendations = []
        for user_id in range(n_users):
            if user_id in test_user_item_map:
                seen_items = set(train_user_item_map.get(user_id, []))
                item_ids, scores = baseline.recommend(
                    user_id, k=k, exclude_seen=True, seen_items=seen_items
                )
                baseline_recommendations.append(item_ids)
            else:
                baseline_recommendations.append([])
        
        all_recommendations[name] = baseline_recommendations
    
    # Prepare ground truth
    ground_truth = []
    for user_id in range(n_users):
        if user_id in test_user_item_map:
            ground_truth.append(set(test_user_item_map[user_id]))
        else:
            ground_truth.append(set())
    
    # Compute metrics
    logger.info("Computing evaluation metrics")
    item_popularity = np.sum(train_interactions, axis=0)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    
    for model_name, recommendations in all_recommendations.items():
        print(f"\n{model_name.upper()} MODEL:")
        print("-" * 50)
        
        results = metrics_calculator.evaluate_all(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k_values=config.evaluation.k_values,
            n_items=n_items,
            item_popularity=item_popularity
        )
        
        metrics_calculator.print_results(results, config.evaluation.k_values)
    
    # Save model
    logger.info("Saving BPR model")
    model_path = Path(config.model_dir) / "bpr_model.pth"
    torch.save({
        'model_state_dict': bpr_model.state_dict(),
        'config': config,
        'n_users': n_users,
        'n_items': n_items,
        'train_losses': bpr_trainer.train_losses,
        'val_losses': bpr_trainer.val_losses
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
