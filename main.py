import os
import torch
import logging
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from vae_model import MolVAE
from train_vae import train_vae
from data_process import OneHotEncoder  # Import the one-hot encoder

# Set up logging
logging.basicConfig(filename='vae_training.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML file
with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def preprocess_smiles(smi):
    """Ensure all backslashes are properly escaped."""
    return smi.replace('\\', '\\\\')

def main():
    # Log start of the process
    logging.info("VAE Training Started")

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(pad_length=config['data']['pad_length'])

    # Define the dataset file path
    dataset_path = config['data']['dataset_path']

    # Check if the npz file exists
    if not os.path.exists(dataset_path):
        logging.info(f"{dataset_path} not found. Loading SMILES data from CSV...")

        # Load SMILES data from CSV
        try:
            smiles_csv_path = './data/250k_zinc_whole.csv'  # Example CSV path
            smiles_df = pd.read_csv(smiles_csv_path)
            smiles_list = smiles_df['SMILES'].tolist()  # Assuming the CSV has a column 'SMILES'
            smiles_list = [preprocess_smiles(smi) for smi in smiles_list]
        except Exception as e:
            logging.error(f"Error loading SMILES from CSV: {e}")
            return

        # One-hot encode the SMILES strings
        logging.info("One-hot encoding SMILES data...")
        one_hot_encoded_smiles = np.array([encoder.one_hot_encode(smi) for smi in smiles_list])

        # Save the one-hot encoded data to .npz for future use
        try:
            np.savez_compressed(dataset_path, arr=one_hot_encoded_smiles)
            logging.info(f"Encoded data saved to {dataset_path}")
        except Exception as e:
            logging.error(f"Error saving encoded data to {dataset_path}: {e}")
            return
    else:
        logging.info(f"Loading encoded data from {dataset_path}...")
        try:
            # Load the pre-encoded SMILES data
            one_hot_encoded_smiles = np.load(dataset_path)['arr'].astype(np.float32)
        except Exception as e:
            logging.error(f"Error loading data from {dataset_path}: {e}")
            return

    # Convert to TensorDataset and create DataLoader
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(one_hot_encoded_smiles))
    train_loader = DataLoader(train_data,
                              batch_size=config['training']['batch_size'],
                              shuffle=True)
    logging.info("Data loaded and ready for training.")
    print(next(iter(train_loader))[0].shape)
    # Model, Optimizer, and Device Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MolVAE(input_size=len(config['data']['charset']),
                   hidden_size=config['model']['hidden_size'],
                   latent_size=config['model']['latent_size'],
                   num_layers=config['model']['num_layers'],
                   gru_output_size=config['model']['gru_output_size']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    logging.info("Model and optimizer initialized.")

    # Train the VAE
    try:
        logging.info("VAE Training Started...")
        trained_model = train_vae(model, train_loader, optimizer, device, epochs=config['training']['epochs'])
        logging.info("VAE Training Completed")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return

    # Save the model
    try:
        torch.save(trained_model.state_dict(), config['data']['save_model_path'])
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


if __name__ == "__main__":
    main()
