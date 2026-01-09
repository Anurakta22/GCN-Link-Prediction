# Graph Convolutional Network for Link Prediction
This project implements a Graph Convolutional Network (GCN)–based link prediction system using Graph Neural Networks (GNNs).
The objective is to predict missing or potential links between nodes in a graph by learning both structural and feature-based representations, supported by quantitative evaluation and visual interpretation of predictions.

## Project Overview
Link prediction is a fundamental problem in graph learning with applications in social networks, recommendation systems, and knowledge graphs.
In this project, a GCN encoder is trained on a real-world graph dataset to learn node embeddings and predict whether a link should exist between two nodes.
To improve interpretability, the project includes visualization of high-confidence predicted links over the original graph structure.

## Key Features
- Graph-based learning: Uses Graph Convolutional Networks (GCNs)
- Link prediction: Predicts missing edges between node pairs
- Binary classification: Link exists vs. link does not exist
- Evaluation metrics: ROC-AUC, Average Precision, Precision, Recall, Precision@K
- Visualization: Interactive graph visualization highlighting predicted links

## Model Architecture
### Encoder
- 2-layer Graph Convolutional Network (GCN)
- Learns node embeddings from graph structure and node features
- ReLU activation with dropout for regularization
### Decoder
- Dot-product decoder
- Computes similarity between node embeddings to score potential links
### Loss Function
- Binary Cross-Entropy Loss
- Trained on both positive and negative edges generated during link splitting
### Dataset
- Cora Citation Network
- Nodes represent research papers
- Edges represent citation relationships
- Node features are bag-of-words representations of paper content
- The dataset is loaded directly using PyTorch Geometric.

## Training & Evaluation
### Data Split
- Train / Validation / Test split using RandomLinkSplit
- Automatic negative sampling to balance positive and negative edges

### Evaluation Metrics
- ROC-AUC
- Average Precision (AP)
- Precision
- Recall
- Precision@K (additional analysis)

### Sample Results
- ROC-AUC: 0.85
- Average Precision: 0.84
- Precision: 0.67
- Recall: 0.91

### Visualization
The project includes a graph visualization where:
- Nodes are shown in blue
- Existing edges are shown in grey
- Top-K predicted links are highlighted in red

This visualization helps interpret how the GCN learns structural patterns and predicts likely missing connections.

## Output & Evaluation
All evaluation metrics and visual outputs are generated within the notebook.
The visualization image of predicted links is saved for reference.

This includes:
- Model training and validation metrics
- Precision and recall evaluation
- Visualization of predicted links over the graph

## How to Use
You can easily run this project using Google Colab by following the steps below.
1. Open the Notebook
- Upload gcn_link_prediction.ipynb to your Google Colab environment.
2. Run the Notebook
- Execute all cells sequentially.
- The notebook installs all required dependencies automatically.
- The model is trained, evaluated, and visualized end-to-end.
