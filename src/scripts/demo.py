"""Streamlit demo for BPR recommendation system."""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.utils import set_seed, get_device
from src.data.dataset import DataGenerator
from src.models.bpr import BPRModel, BPRTrainer
from src.models.baselines import (
    PopularityBaseline, UserKNNBaseline, ItemKNNBaseline, RandomBaseline
)
from src.utils.metrics import RecommendationMetrics


@st.cache_data
def load_data_and_models():
    """Load data and trained models."""
    # Load configuration
    config = Config.from_yaml("configs/default.yaml")
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Generate data
    data_generator = DataGenerator(config)
    interactions, user_item_map, item_user_map = data_generator.generate_interactions()
    
    # Split data
    train_interactions, val_interactions, test_interactions, train_user_item_map, val_user_item_map = data_generator.split_data(
        interactions, user_item_map
    )
    
    # Initialize models
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
        device="cpu"  # Use CPU for demo
    )
    
    # Train models quickly for demo
    from src.data.dataset import BPRDataset
    from torch.utils.data import DataLoader
    
    train_dataset = BPRDataset(train_interactions, train_user_item_map, seed=config.random_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
    
    # Train BPR model (few epochs for demo)
    bpr_trainer.train(train_dataloader, num_epochs=20, verbose=False)
    
    # Baseline models
    baselines = {
        "Popularity": PopularityBaseline(),
        "User KNN": UserKNNBaseline(k=50),
        "Item KNN": ItemKNNBaseline(k=50),
        "Random": RandomBaseline(seed=config.random_seed)
    }
    
    for baseline in baselines.values():
        baseline.fit(train_interactions)
    
    return {
        'config': config,
        'interactions': interactions,
        'train_interactions': train_interactions,
        'test_interactions': test_interactions,
        'train_user_item_map': train_user_item_map,
        'test_user_item_map': {user_id: np.where(test_interactions[user_id] > 0)[0].tolist() 
                              for user_id in range(test_interactions.shape[0]) 
                              if np.sum(test_interactions[user_id]) > 0},
        'bpr_trainer': bpr_trainer,
        'baselines': baselines,
        'n_users': n_users,
        'n_items': n_items
    }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="BPR Recommendation System Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Bayesian Personalized Ranking (BPR) Recommendation System")
    st.markdown("---")
    
    # Load data and models
    with st.spinner("Loading data and training models..."):
        data = load_data_and_models()
    
    st.success("Models loaded successfully!")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # User selection
    user_id = st.sidebar.selectbox(
        "Select User",
        options=list(range(data['n_users'])),
        format_func=lambda x: f"User {x}"
    )
    
    # Number of recommendations
    k = st.sidebar.slider("Number of Recommendations", 5, 50, 10)
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=["BPR", "Popularity", "User KNN", "Item KNN", "Random"]
    )
    
    # Get user's interaction history
    user_interactions = data['train_user_item_map'].get(user_id, [])
    user_test_items = data['test_user_item_map'].get(user_id, [])
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("User Profile")
        st.write(f"**User ID:** {user_id}")
        st.write(f"**Training Interactions:** {len(user_interactions)} items")
        st.write(f"**Test Interactions:** {len(user_test_items)} items")
        
        if user_interactions:
            st.write("**Training Items:**")
            st.write(f"Items: {user_interactions}")
        
        if user_test_items:
            st.write("**Test Items (Ground Truth):**")
            st.write(f"Items: {user_test_items}")
    
    with col2:
        st.subheader("Recommendations")
        
        # Get recommendations
        if model_name == "BPR":
            seen_items = set(user_interactions)
            item_ids, scores = data['bpr_trainer'].get_recommendations(
                user_id, k=k, exclude_seen=True, seen_items=seen_items
            )
            recommendations = item_ids.tolist()
            recommendation_scores = scores.tolist()
        else:
            baseline = data['baselines'][model_name]
            seen_items = set(user_interactions)
            recommendations, recommendation_scores = baseline.recommend(
                user_id, k=k, exclude_seen=True, seen_items=seen_items
            )
        
        # Display recommendations
        if recommendations:
            st.write(f"**Top-{k} Recommendations:**")
            
            # Create DataFrame for display
            rec_df = pd.DataFrame({
                'Item ID': recommendations,
                'Score': [f"{score:.4f}" for score in recommendation_scores],
                'In Test Set': ['✅' if item in user_test_items else '❌' for item in recommendations]
            })
            
            st.dataframe(rec_df, use_container_width=True)
            
            # Calculate hit rate
            hits = sum(1 for item in recommendations if item in user_test_items)
            hit_rate = hits / len(user_test_items) if user_test_items else 0
            st.metric("Hit Rate", f"{hit_rate:.2%}")
        else:
            st.warning("No recommendations available for this user.")
    
    # Evaluation section
    st.markdown("---")
    st.subheader("Model Evaluation")
    
    # Calculate metrics for all models
    metrics_calculator = RecommendationMetrics()
    
    # Prepare ground truth
    ground_truth = []
    for uid in range(data['n_users']):
        if uid in data['test_user_item_map']:
            ground_truth.append(set(data['test_user_item_map'][uid]))
        else:
            ground_truth.append(set())
    
    # Get recommendations for all models
    all_recommendations = {}
    max_k = 50
    
    # BPR recommendations
    bpr_recommendations = []
    for uid in range(data['n_users']):
        if uid in data['test_user_item_map']:
            seen_items = set(data['train_user_item_map'].get(uid, []))
            item_ids, scores = data['bpr_trainer'].get_recommendations(
                uid, k=max_k, exclude_seen=True, seen_items=seen_items
            )
            bpr_recommendations.append(item_ids.tolist())
        else:
            bpr_recommendations.append([])
    
    all_recommendations["BPR"] = bpr_recommendations
    
    # Baseline recommendations
    for name, baseline in data['baselines'].items():
        baseline_recommendations = []
        for uid in range(data['n_users']):
            if uid in data['test_user_item_map']:
                seen_items = set(data['train_user_item_map'].get(uid, []))
                item_ids, scores = baseline.recommend(
                    uid, k=max_k, exclude_seen=True, seen_items=seen_items
                )
                baseline_recommendations.append(item_ids)
            else:
                baseline_recommendations.append([])
        
        all_recommendations[name] = baseline_recommendations
    
    # Compute metrics
    k_values = [5, 10, 20]
    item_popularity = np.sum(data['train_interactions'], axis=0)
    
    # Create metrics comparison
    metrics_data = []
    
    for model_name, recommendations in all_recommendations.items():
        results = metrics_calculator.evaluate_all(
            recommendations=recommendations,
            ground_truth=ground_truth,
            k_values=k_values,
            n_items=data['n_items'],
            item_popularity=item_popularity
        )
        
        for k in k_values:
            metrics_data.append({
                'Model': model_name,
                'K': k,
                'Precision': results['precision'][k],
                'Recall': results['recall'][k],
                'NDCG': results['ndcg'][k],
                'Hit Rate': results['hit_rate'][k]
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Precision@K Comparison**")
        fig_precision = px.bar(
            metrics_df, x='K', y='Precision', color='Model',
            title='Precision@K by Model',
            barmode='group'
        )
        st.plotly_chart(fig_precision, use_container_width=True)
    
    with col2:
        st.write("**NDCG@K Comparison**")
        fig_ndcg = px.bar(
            metrics_df, x='K', y='NDCG', color='Model',
            title='NDCG@K by Model',
            barmode='group'
        )
        st.plotly_chart(fig_ndcg, use_container_width=True)
    
    # Coverage and diversity metrics
    st.write("**Coverage and Diversity Metrics**")
    
    coverage_data = []
    diversity_data = []
    
    for model_name, recommendations in all_recommendations.items():
        coverage = metrics_calculator.coverage(recommendations, data['n_items'])
        diversity = metrics_calculator.diversity(recommendations)
        
        coverage_data.append({'Model': model_name, 'Coverage': coverage})
        diversity_data.append({'Model': model_name, 'Diversity': diversity})
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_coverage = px.bar(
            pd.DataFrame(coverage_data), x='Model', y='Coverage',
            title='Catalog Coverage by Model'
        )
        st.plotly_chart(fig_coverage, use_container_width=True)
    
    with col2:
        fig_diversity = px.bar(
            pd.DataFrame(diversity_data), x='Model', y='Diversity',
            title='Diversity by Model'
        )
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    # Dataset statistics
    st.markdown("---")
    st.subheader("Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", data['n_users'])
        st.metric("Total Items", data['n_items'])
    
    with col2:
        st.metric("Training Interactions", np.sum(data['train_interactions']))
        st.metric("Test Interactions", np.sum(data['test_interactions']))
    
    with col3:
        density = np.mean(data['interactions'])
        st.metric("Interaction Density", f"{density:.3f}")
        
        # Popularity bias
        from src.utils.utils import calculate_popularity_bias
        bias = calculate_popularity_bias(data['interactions'])
        st.metric("Popularity Bias (Gini)", f"{bias:.3f}")


if __name__ == "__main__":
    main()
