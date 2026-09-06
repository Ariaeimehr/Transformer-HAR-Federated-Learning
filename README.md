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

    pip install tensorflow numpy pandas scikit-learn matplotlib seaborn


## Quick Start / Usage
MHAT-FL is highly modular. Below is a basic example of how to initialize the framework and run a federated training loop using the **FedPer** strategy:

    from Data_Preprocessing_and_Windowing import sliding_window
    from Federated_Learning_Utilities import create_clients, batch_data, weight_scalling_factor, scale_model_weights, sum_scaled_weights
    from transformer import Transformer 

    # 1. Initialize the global MHAT model
    global_model = Transformer
    global_weights = global_model.get_weights()

    # 2. Simulate edge clients and distribute sensor data
    clients = create_clients(X_train, y_train, num_clients=4)
    clients_batched = {name: batch_data(data) for name, data in clients.items()}

    # 3. Execute the Federated Training Loop (FedPer Strategy)
    comms_round = 100
    for round in range(comms_round):
        scaled_local_weights = []
        
        for client in clients.keys():
            local_model = Transformer
            local_weights = local_model.get_weights()
            
            # Isolate final dense layers for personalization (FedPer)
            local_weights[:-1] = global_weights[:-1]
            local_model.set_weights(local_weights)
            
            # Perform local training
            local_model.fit(clients_batched[client], epochs=1, verbose=0)
            
            # Scale and collect updated weights
            scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weights.append(scaled_weights)
            
        # Secure global aggregation 
        average_weights = sum_scaled_weights(scaled_local_weights)
        global_model.set_weights(average_weights)


## Citation
If you use this framework in your research, please cite our upcoming paper:

    @article{ariaeimehr2026mhatfl,
      title={MHAT-FL: An Open-Source TensorFlow Framework for Federated Human Activity Recognition with Attention-Matrix Positional Encoding},
      author={Ariaeimehr, Mohammad},
      journal={arXiv preprint},
      year={2026}
    }
