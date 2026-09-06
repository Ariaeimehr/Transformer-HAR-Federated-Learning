# MHAT-FL: Federated Learning for Human Activity Recognition using Attention-Matrix Positional Encoding

This repository contains the official open-source implementation of **MHAT-FL**, a privacy-preserving Federated Learning (FL) framework specifically designed for Human Activity Recognition (HAR) using customized Transformer architectures.

## Overview
MHAT-FL aims to classify human physical activities from high-frequency time-series sensor data without centralizing sensitive user information. By leveraging a federated approach, local edge devices train the model independently, and only the updated parameters are aggregated globally. This framework entirely bypasses heavy third-party FL wrappers, providing a transparent, native TensorFlow implementation.

## Key Innovations & Features
* **Attention-Matrix Positional Encoding:** Unlike standard architectures, MHAT-FL injects positional encoding directly into the Query and Key matrices at each attention head, significantly improving the capture of complex temporal dependencies in multivariate sensor data.
* **Transparent Federated Engine:** Implements distributed training loops natively from scratch, allowing seamless switching between Federated Averaging (**FedAvg**) and Federated Personalization (**FedPer**).
* **Privacy & Security:** Features built-in evaluation pipelines to assess model vulnerability against Membership Inference Attacks (MIA).
* **Data Processing Pipeline:** Includes automated sliding window mechanisms and data sharding to simulate real-world non-IID client environments efficiently.

## Repository Structure
* `Data_Preprocessing_and_Windowing.py`: Handles sliding window segmentation, feature extraction, and train/test splits.
* `Imports_and_Setup.py`: Core library configurations and environment setup.
* `Federated_Learning_Utilities.py`: Client simulation, non-IID data batching, and custom weight scaling/aggregation algorithms.
* `Federated_Training_Loop.py`: The main execution script managing communication rounds, local training, and global model updates.
* `Model_Evaluation_and_Visualization.py`: Calculates macro/weighted performance metrics (Precision, Recall, F1-Score) and plots the Confusion Matrix.
* `transformer.py`: The core MHAT architecture with custom Multi-Head Attention layers.

## Prerequisites
Ensure you have the following dependencies installed in your Python environment:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
