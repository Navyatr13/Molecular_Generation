import torch
import numpy as np
from models.vae_model import MolVAE
from scripts.data_process import OneHotFeaturizer
from utils import decode_and_validate_molecules, save_results
import yaml


def load_pretrained_model(config, device):
    """Load the pre-trained VAE model."""
    model = MolVAE().to(device)
    model.load_state_dict(torch.load(config['training']['checkpoint_path']))
    model.eval()
    return model


def sample_molecules(model, smiles_in, config, device):
    """Sample molecular structures from VAE using the latent space."""
    # One-hot encode the SMILES input
    ohf = OneHotFeaturizer(config['data']['charset'], config['data']['pad_length'])
    oh_smiles = ohf.featurize(smiles_in).astype(np.float32)

    targetset = torch.utils.data.TensorDataset(torch.from_numpy(oh_smiles))
    target_loader = torch.utils.data.DataLoader(targetset, batch_size=len(smiles_in), shuffle=False)

    # Pass SMILES through VAE to generate latent space and reconstructed molecules
    for _, data in enumerate(target_loader):
        data = data[0].transpose(1, 2).to(device)
        recon_batch, _, _, latent_space = model(data)

    return recon_batch, latent_space, ohf


def main():
    """Main function to load the model, sample molecules, and save results."""
    # Load configuration
    with open('./configs/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model
    model = load_pretrained_model(config, device)

    # Input SMILES for sampling
    smiles_in = []
    with open(config['data']['dataset_path'], "r") as file:
        for line in file:
            smiles_in.append(line.strip('\n'))

    # Sample molecules using the VAE
    recon_batch, latent_space, ohf = sample_molecules(model, smiles_in, config, device)

    # Decode and validate molecules
    mol_list, mol_list_val = decode_and_validate_molecules(recon_batch, ohf, config)

    # Save results
    save_results(mol_list, mol_list_val)

    # Save latent space for target dataset
    torch.save(latent_space, 'target_latent_2.pt')


if __name__ == "__main__":
    main()
