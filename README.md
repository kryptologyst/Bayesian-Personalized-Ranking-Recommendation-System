# Bayesian Personalized Ranking Recommendation System

A production-ready implementation of the Bayesian Personalized Ranking algorithm for implicit feedback recommendation systems.

## Overview

This project implements a complete BPR recommendation system with:

- **BPR Model**: Core Bayesian Personalized Ranking implementation with proper pairwise loss
- **Baseline Models**: Popularity, User-KNN, Item-KNN, and Random baselines for comparison
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Diversity, and Novelty metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Clean code structure, type hints, comprehensive documentation, and testing

## Features

### Core Algorithm
- Bayesian Personalized Ranking (BPR) with proper pairwise ranking loss
- Matrix factorization with user and item embeddings
- L2 regularization for preventing overfitting
- Stochastic gradient descent optimization

### Baseline Comparisons
- **Popularity**: Recommends most popular items
- **User-KNN**: User-based collaborative filtering
- **Item-KNN**: Item-based collaborative filtering  
- **Random**: Random recommendations for comparison

### Evaluation Metrics
- **Ranking Metrics**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K
- **Diversity Metrics**: Catalog coverage, intra-list diversity
- **Novelty Metrics**: Average novelty of recommendations
- **Bias Metrics**: Popularity bias analysis

### Interactive Demo
- User-friendly Streamlit interface
- Real-time recommendation generation
- Model comparison visualizations
- Dataset statistics and analysis

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kryptologyst/Bayesian-Personalized-Ranking-Recommendation-System.git
cd Bayesian-Personalized-Ranking-Recommendation-System
```

2. **Install dependencies**:
```bash
pip install -e .
```

Or install with optional dependencies:
```bash
pip install -e ".[dev,mlflow,wandb]"
```

3. **Verify installation**:
```bash
python -c "import src; print('Installation successful!')"
```

## Quick Start

### 1. Training the Model

Train the BPR model with baseline comparisons:

```bash
python src/scripts/train.py --config configs/default.yaml --verbose
```

### 2. Running the Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run src/scripts/demo.py
```

The demo will open in your browser at `http://localhost:8501`.

### 3. Configuration

Modify `configs/default.yaml` to adjust:

- **Model parameters**: embedding dimension, learning rate, regularization
- **Data parameters**: number of users/items, interaction probability
- **Training parameters**: epochs, batch size, early stopping
- **Evaluation parameters**: k-values, metrics to compute

## Project Structure

```
0344_Bayesian_personalized_ranking/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   └── dataset.py           # Dataset classes and data loaders
│   ├── models/                   # Model implementations
│   │   ├── bpr.py               # BPR model and trainer
│   │   └── baselines.py         # Baseline models
│   ├── utils/                    # Utilities
│   │   ├── config.py            # Configuration management
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── utils.py             # Helper functions
│   └── scripts/                  # Executable scripts
│       ├── train.py             # Training script
│       └── demo.py               # Streamlit demo
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── data/                         # Data directory
├── models/                       # Saved models
├── logs/                         # Log files
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── assets/                       # Static assets
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Usage Examples

### Basic Training

```python
from src.utils.config import Config
from src.data.dataset import DataGenerator
from src.models.bpr import BPRModel, BPRTrainer

# Load configuration
config = Config.from_yaml("configs/default.yaml")

# Generate data
data_generator = DataGenerator(config)
interactions, user_item_map, _ = data_generator.generate_interactions()

# Initialize model
model = BPRModel(
    n_users=interactions.shape[0],
    n_items=interactions.shape[1],
    embedding_dim=config.model.embedding_dim
)

# Train model
trainer = BPRTrainer(model, learning_rate=config.model.learning_rate)
trainer.train(train_dataloader, num_epochs=config.model.num_epochs)
```

### Getting Recommendations

```python
# Get top-10 recommendations for user 0
item_ids, scores = trainer.get_recommendations(
    user_id=0, 
    k=10, 
    exclude_seen=True, 
    seen_items=set(user_item_map[0])
)

print(f"Recommended items: {item_ids}")
print(f"Scores: {scores}")
```

### Model Evaluation

```python
from src.utils.metrics import RecommendationMetrics

# Initialize metrics calculator
metrics = RecommendationMetrics()

# Evaluate recommendations
results = metrics.evaluate_all(
    recommendations=all_recommendations,
    ground_truth=ground_truth,
    k_values=[5, 10, 20],
    n_items=n_items
)

# Print results
metrics.print_results(results, [5, 10, 20])
```

## Configuration

The system uses YAML configuration files. Key parameters:

### Model Configuration
```yaml
model:
  embedding_dim: 64        # Dimension of user/item embeddings
  learning_rate: 0.01     # Learning rate for optimizer
  regularization: 0.01    # L2 regularization strength
  batch_size: 1024        # Training batch size
  num_epochs: 100         # Number of training epochs
  early_stopping_patience: 10  # Early stopping patience
```

### Data Configuration
```yaml
data:
  n_users: 1000           # Number of users
  n_items: 500            # Number of items
  interaction_probability: 0.1  # Probability of user-item interaction
  test_ratio: 0.2         # Fraction of data for testing
  val_ratio: 0.1          # Fraction of data for validation
```

## Evaluation Metrics

### Ranking Metrics
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation

### Diversity Metrics
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity of recommendations

### Novelty Metrics
- **Novelty**: Average novelty of recommended items (based on popularity)

## Advanced Features

### Custom Data Loading
```python
# Load your own interaction data
interactions = np.load("your_interactions.npy")
user_item_map = {user_id: item_list for user_id, item_list in your_data}

# Create dataset
dataset = BPRDataset(interactions, user_item_map)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

### Model Persistence
```python
# Save trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'n_users': n_users,
    'n_items': n_items
}, 'models/bpr_model.pth')

# Load model
checkpoint = torch.load('models/bpr_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Hyperparameter Tuning
```python
# Try different configurations
configs = [
    {"embedding_dim": 32, "learning_rate": 0.01},
    {"embedding_dim": 64, "learning_rate": 0.005},
    {"embedding_dim": 128, "learning_rate": 0.01}
]

for config_params in configs:
    config.model.embedding_dim = config_params["embedding_dim"]
    config.model.learning_rate = config_params["learning_rate"]
    # Train and evaluate...
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff check src/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. UAI.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. ICDM.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- Scikit-learn team for machine learning utilities
# Bayesian-Personalized-Ranking-Recommendation-System
