import torch
import yaml
import numpy as np
import pandas as pd

from models.vae_model import MolVAE
from scripts.data_process import OneHotEncoder
from utils.utils import decode_and_validate_molecules, save_results, preprocess_smiles, load_pretrained_model, sample_molecules
from torch.utils.data import DataLoader, TensorDataset, random_split
from scripts.train_vae import loss_function, fine_tune_vae, evaluate_vae


def main():
    """Main function to load the model, sample molecules, and save results."""
    # Load configuration
    with open('./configs/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model
    model = load_pretrained_model(config, device)
    smiles_csv_path = config['data']['target_dataset_path']

    smiles_df = pd.read_csv(smiles_csv_path)
    smiles_list = smiles_df['SMILES'].tolist() 
    smiles_list = [preprocess_smiles(smi) for smi in smiles_list]
    
    if config['training']['fine_tune']:
        # One-hot encode the EGFR SMILES data
        encoder = OneHotEncoder(pad_length=config['data']['pad_length'])
        one_hot_encoded_smiles = np.array([encoder.one_hot_encode(smi) for smi in smiles_list], dtype=np.float32)
        one_hot_encoded_smiles = torch.from_numpy(one_hot_encoded_smiles)

        # Split into training and validation sets (e.g., 80% train, 20% validation)
        train_size = int(0.8 * len(one_hot_encoded_smiles))
        val_size = len(one_hot_encoded_smiles) - train_size
        train_data, val_data = random_split(one_hot_encoded_smiles, [train_size, val_size])
        
        # Now create TensorDataset from the Subset
        train_data = TensorDataset(train_data.dataset[train_data.indices])
        val_data = TensorDataset(val_data.dataset[val_data.indices])
    
        # Create DataLoader for training and validation sets
        train_loader = DataLoader(train_data,
                                  batch_size=  config['training']['batch_size'],
                                  shuffle=True, num_workers = 4)
    
        val_loader = DataLoader(val_data,
                                batch_size= config['training']['batch_size'],
                                shuffle=False, num_workers = 4)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['fine_tune_learning_rate'])
    
        # Fine-tune the model on the EGFR dataset
        fine_tuned_model = fine_tune_vae(model, train_loader, val_loader, optimizer, device, config)
        recon_batch, latent_space, ohf = sample_molecules(fine_tuned_model, smiles_list, config, device)
    
        # Decode and validate molecules
        mol_list, mol_list_val = decode_and_validate_molecules(recon_batch, ohf, config)
        save_results(mol_list, mol_list_val)
    
        # Save latent space for target dataset
        torch.save(latent_space, 'egfr_latent_after_fine_tuning.pt')
    
    
    
    else:
        # Sample molecules using the VAE
        recon_batch, latent_space, ohf = sample_molecules(model, smiles_list, config, device)
    
        # Decode and validate molecules
        mol_list, mol_list_val = decode_and_validate_molecules(recon_batch, ohf, config)
    
        # Save results
        save_results(mol_list, mol_list_val)
    
        # Save latent space for target dataset
        torch.save(latent_space, 'target_latent_without_finetuning.pt')


if __name__ == "__main__":
    main()
