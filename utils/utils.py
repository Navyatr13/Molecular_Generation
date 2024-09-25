# utils.py

import csv
import pandas as pd
import torch

from rdkit import Chem
from models.vae_model import MolVAE


def decode_and_validate_molecules(recon_batch, ohf, config):
    """Decode the one-hot reconstructions back into SMILES and validate them."""
    recon_batch = recon_batch.cpu().detach().numpy()
    mol_list = []
    mol_list_val = []
    invalids = 0
    valids = 0

    for l in range(recon_batch.shape[0]):
        y = np.argmax(recon_batch[l], axis=1)
        vec = ohf.decode_smiles_from_index(y)
        mol = Chem.MolFromSmiles(vec)
        mol_list.append(vec)

        if not mol:
            invalids += 1
            mol_list_val.append([vec, 'Invalid'])
        else:
            valids += 1
            mol_list_val.append([vec, 'Valid'])

    print(f"Valid: {valids}, Invalid: {invalids}")

    return mol_list, mol_list_val


def save_results(mol_list, mol_list_val):
    """Save the decoded molecules and their validity to a CSV file."""
    with open('mol_decoded.csv', 'w') as smiles_file:
        wr = csv.writer(smiles_file, delimiter="\n")
        wr.writerow(mol_list)

    df = pd.DataFrame(mol_list_val, columns=['Mols', 'Validity'])
    df.to_csv('mol_validity_decoded.csv')

def preprocess_smiles(smi):
    """Ensure all backslashes are properly escaped."""
    return smi.replace('\\', '\\\\')
    
def load_pretrained_model(config, device):
    """Load the pre-trained VAE model for fine-tuning."""
    model = MolVAE(input_size=len(config['data']['charset']),
                   hidden_size=config['model']['hidden_size'],
                   latent_size=config['model']['latent_size'],
                   num_layers=config['model']['num_layers'],
                   gru_output_size=config['model']['gru_output_size']).to(device)
    
    # Load the pre-trained weights from ZINC training
    checkpoint = torch.load(config['training']['checkpoint_path'], map_location=device, weights_only = True)
    model.load_state_dict(checkpoint)
    if config['training']['fine_tune']:
    
        model.train()  # Set model to training mode 
    else:
        model.eval()
    return model
    
def sample_molecules(model, smiles_in, config, device):
    """Sample molecular structures from VAE using the latent space."""
    dataset_path = config['data']['target_saved_features']
    if not os.path.exists(dataset_path):
        # One-hot encode the SMILES input
        ohf = OneHotEncoder(pad_length=config['data']['pad_length'])
        one_hot_encoded_smiles = []
    
        for smi in smiles_in:
            try:
                # One-hot encode and append
                encoded = ohf.one_hot_encode(smi)
                one_hot_encoded_smiles.append(encoded)
            except Exception as e:
                print(f"Error encoding SMILES {smi}: {e}")
    else:
        one_hot_encoded_smiles = np.load(dataset_path)['arr'].astype(np.float32)
        
    one_hot_encoded_smiles =torch.from_numpy(np.array(one_hot_encoded_smiles),dtype=float32)
    targetset = torch.utils.data.TensorDataset(one_hot_encoded_smiles)
    target_loader = torch.utils.data.DataLoader(targetset, batch_size=len(smiles_in), shuffle=False)

    # Pass SMILES through VAE to generate latent space and reconstructed molecules
    for _, data in enumerate(target_loader):
        data = data[0].transpose(1, 2).to(device)
        recon_batch, _, _, latent_space = model(data)

    return recon_batch, latent_space, ohf