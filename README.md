# Realtime Fashion Recommender

A machine learning-powered fashion recommendation system built with TensorFlow Recommenders and Hopsworks, featuring real-time inference and interactive customer experiences. This project implements a two-tower retrieval model combined with a ranking model to provide personalized fashion recommendations based on H&M customer data.

## 🌟 Features

- **Real-time Recommendations**: Get personalized fashion item recommendations with real-time inference
- **Two-Tower Architecture**: Implements customer and item embedding towers for efficient candidate retrieval
- **Ranking Model**: Uses CatBoost for ranking and reordering recommended items
- **Interactive UI**: Streamlit-based web interface for customer interaction and feedback
- **MLOps Pipeline**: Complete feature engineering, training, and deployment pipeline using Hopsworks
- **Real-time Feature Store**: Track customer interactions and update recommendations dynamically

## 🏗️ Architecture

The system consists of several key components:

1. **Feature Pipeline**: Processes H&M customer, article, and transaction data
2. **Two-Tower Model**: Retrieval model that learns customer and item embeddings
3. **Ranking Model**: CatBoost model for ranking candidate items
4. **Feature Store**: Hopsworks-based feature management and serving
5. **Streamlit App**: Interactive UI for customer recommendations and feedback

## 📊 Models

### Two-Tower Retrieval Model
- **Customer Tower**: Encodes customer features (age, membership status, interaction history)
- **Item Tower**: Encodes article features (garment type, color, department, embeddings)
- **Architecture**: TensorFlow/Keras with embedding layers and feed-forward networks

### Ranking Model
- **Algorithm**: CatBoost gradient boosting
- **Features**: Customer demographics, item attributes, interaction patterns
- **Purpose**: Rerank candidate items for personalized recommendations

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Hopsworks account and API key
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/haturusinghe/realtime-fashion-recommender.git
cd realtime-fashion-recommender
```

2. Install Python and dependencies using the Makefile:
```bash
# Install Python 3.11 using uv
make install-python

# Create virtual environment and install all dependencies
make install
```

3. Set up environment variables:
```bash
# Create .env file with your Hopsworks credentials
echo "HOPSWORKS_API_KEY=your_api_key_here" > .env
```

### Quick Start with Makefile

The project includes a convenient Makefile for easy execution of all pipeline steps:

#### Complete ML Pipeline
```bash
# 1. Feature engineering
make feature-engineering

# 2. Train retrieval model
make train-retrieval

# 3. Train ranking model  
make train-ranking

# 4. Generate item embeddings
make create-embeddings

# 5. Create model deployments
make create-deployments

# 6. Start the Streamlit UI
make start-ui
```

#### Individual Steps
```bash
# Run feature engineering pipeline
make feature-engineering

# Train the two-tower retrieval model
make train-retrieval

# Train the CatBoost ranking model
make train-ranking

# Compute item embeddings for similarity search
make create-embeddings

# Deploy models to Hopsworks
make create-deployments

# Start the interactive web application
make start-ui
```


## 📁 Project Structure

```
recsys/
├── config.py                 # Configuration settings
├── features/                 # Feature engineering modules
├── hopsworks_integration/    # Hopsworks feature store integration
├── raw_data_sources/         # Data extraction from H&M dataset
├── training/                 # Model training modules
└── ui/                       # Streamlit UI components

notebooks/                    # Jupyter notebooks for pipeline execution
Makefile                     # Automation commands for easy project execution
streamlit_app.py             # Main Streamlit application
pyproject.toml              # Python project configuration and dependencies
```



## 🛠️ Technologies Used

- **Package Manager**: uv (fast Python package manager)
- **ML Frameworks**: TensorFlow, TensorFlow Recommenders, CatBoost
- **Feature Store**: Hopsworks
- **Data Processing**: Polars, Pandas
- **UI**: Streamlit
- **Embeddings**: Sentence Transformers
- **Deployment**: Hopsworks Model Serving
- **Environment Management**: Python 3.11, virtual environments

## 📈 Data

The system uses the H&M Personalized Fashion Recommendations dataset from Kaggle

