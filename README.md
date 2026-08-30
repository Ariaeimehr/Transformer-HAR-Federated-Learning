# Federated Learning for Human Activity Recognition using Transformer Networks

This repository implements a privacy-preserving Federated Learning (FL) framework for Human Activity Recognition (HAR) using Transformer architectures.

## Overview
The project aims to classify human physical activities from time-series sensor data without centralizing sensitive user information. By leveraging a federated approach, local devices train the Transformer model independently, and only the updated parameters (weights) are aggregated globally.

## Key Features
* **Transformer Architecture:** Incorporates Positional Encoding and Attention mechanisms to capture complex temporal dependencies in multivariate time-series data.
* **Federated Learning:** Implements distributed training loops using aggregation strategies to keep raw data localized on client nodes.
* **Privacy & Security:** Features Differential Privacy (Gaussian noise injection) and evaluates model vulnerability against Membership Inference Attacks (MIA).
* **Data Processing:** Automated sliding window mechanisms for efficient sensor data segmentation.

## Repository Structure
* `Data Preprocessing & Windowing.py`: Handles sliding window segmentation, feature extraction, and train/test splits.
* `Imports & Setup.py`: Core library configurations and environment setup.
* `Federated Learning Utilities.py`: Client simulation, data batching, and weight scaling algorithms.
* `Federated Training Loop.py`: The main execution script managing communication rounds and global model updates.
* `Model Evaluation & Visualization.py`: Calculates performance metrics (Precision, Recall, F1-Score) and plots the Confusion Matrix.

## Prerequisites
Ensure you have the following dependencies installed in your Python environment:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
