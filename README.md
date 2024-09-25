# Variational Autoencoder for Molecular Generation

This repository contains the implementation of a Variational Autoencoder (VAE) designed for generating molecular structures based on SMILES strings. The model encodes SMILES strings into a latent space and decodes them to generate new molecular structures. The dataset used includes 250k molecules from the ZINC dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)

## Overview

The goal of this project is to use a Variational Autoencoder (VAE) to generate new molecules by training on SMILES strings. It utilizes PyTorch and RDKit for molecular handling and PyTorch's deep learning framework for building and training the VAE model.

## Features

- **SMILES-based molecular generation**: Encode and decode molecular structures using SMILES strings.
- **One-hot encoding of SMILES strings** for input to the VAE model.
- **Configurable model architecture** using YAML configuration files.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Navyatr13/Molecular_Generation.git
cd Molecular_Generation
```

### 2. Install dependencies

```commandline
pip install -r requirements.txt
```
### 3. Dataset
The model is trained on a dataset of molecules from the ZINC database. 
Ensure that the dataset is available in the ./data/ folder. 
You can download a dataset like the 250k random subset of ZINC.

### 4. Usage
Training the VAE
To train the model, use the following command:
```commandline
python main.py
```

###  Configuration
You can adjust hyperparameters like batch size, learning rate, number of epochs, and model architecture through the configs/config.yaml file:
    batch_size: 256 
    learning_rate: 0.001
    epochs: 100
    model:
        hidden_size: 512
        latent_size: 128
        num_layers: 3
### Model Output
After training, the model weights will be saved in the ./save/ directory, and the training logs can be found in the vae_training.log file.

### Evaluating the Model
Once the model is trained, you can generate new molecular structures by decoding samples from the latent space. Further evaluation or visualization methods can be added depending on your needs.

### Model Architecture
1. Encoder: The encoder maps the one-hot encoded SMILES string into a latent space using a series of convolutional layers and fully connected layers.
2. Decoder: The decoder takes the latent vector and generates a new SMILES string using a GRU-based architecture.
3. Latent Space: The latent space representation allows for generating new molecular structures by sampling from the distribution.
### Training
The training loop is implemented using PyTorch with the tqdm library to track progress. The model is trained with a combination of a reconstruction loss (binary cross-entropy) and a KL divergence loss.

### Example Output
```
Epoch 1/100: 100%|██████████| 500/500 [00:20<00:00, 24.50batch/s, loss=0.056]
End of Epoch 1: Train Loss 0.0456 
```
### Sampling Molecular Structures
Once the VAE model is trained, you can sample new molecular structures using the sample.py script. This script uses the pre-trained VAE model to generate new molecules by decoding latent space representations.

To sample molecules, run:
``` 
python sample.py
```
This will generate new molecular structures based on the input SMILES strings provided in the dataset (egfr_targetset.smi). The script decodes the sampled latent vectors into SMILES strings and validates them using RDKit.

1. Input: The input SMILES strings are located in the data/ directory (e.g., egfr_targetset.smi).
2. Output: The generated SMILES strings and their validity (valid/invalid molecules) will be saved in two CSV files:
mol_decoded.csv (list of generated SMILES)
mol_validity_decoded.csv (list of generated SMILES along with validity)
3. Latent Space: The latent space representation of the target dataset is saved in target_latent.pt.