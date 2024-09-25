# utils.py

import csv
import pandas as pd
from rdkit import Chem


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
