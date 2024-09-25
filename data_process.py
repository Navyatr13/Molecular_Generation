import numpy as np

# Define the charset for SMILES one-hot encoding
CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
           '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
           'c', 'l', 'n', 'o', 'r', 's']
class OneHotEncoder:
    def __init__(self, charset=CHARSET, pad_length=120):
        self.charset = charset
        self.pad_length = pad_length

    def one_hot_array(self, i):
        """Convert index to one-hot array."""
        return [int(ix == i) for ix in range(len(self.charset))]

    def one_hot_index(self, c):
        """Find the index of character in the charset."""
        try:
            return self.charset.index(c)
        except ValueError:
            print(f"Error: Character '{c}' not found in charset!")  # Debugging: Print the character causing the issue
            raise

    def pad_smi(self, smi):
        """Pad the SMILES string to the fixed length."""
        padded_smi = smi.ljust(self.pad_length)
        return padded_smi

    def one_hot_encode(self, smi):
        """Convert SMILES string into one-hot encoded representation."""
        smi = self.pad_smi(smi)  # Pad the SMILES string to the required length
        return np.array([
            self.one_hot_array(self.one_hot_index(char)) for char in smi
        ])


#Helper function to check if there are any missing characters in the CHARSET
def check_missing_chars(smiles_list, charset):
    unique_chars = set(''.join(smiles_list))
    missing_chars = unique_chars - set(charset)

    if missing_chars:
        print(f"Missing characters: {missing_chars}")
    else:
        print("No missing characters in charset.")
