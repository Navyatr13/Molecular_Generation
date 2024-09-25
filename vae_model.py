import torch
import torch.nn as nn
import torch.nn.functional as F

class MolVAE(nn.Module):
    """Variational Autoencoder for SMILES Molecular Data."""

    def __init__(self, input_size, hidden_size, latent_size, num_layers, gru_output_size):
        super(MolVAE, self).__init__()

        # Encoder
        self.conv1d1 = nn.Conv1d(input_size, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc0 = nn.Linear(940, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.gru = nn.GRU(latent_size, gru_output_size, num_layers, batch_first=True)
        self.fc3 = nn.Linear(gru_output_size, input_size)

    def encode(self, x):
        """Encoder to produce mean and log-variance."""
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparametrize(self, mean, logv):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z):
        """Decoder to reconstruct data from latent space."""
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(out.size(0), out.size(1), -1)
        return F.softmax(self.fc3(out_reshape), dim=1)

    def forward(self, x):
        """Forward pass through the model."""
        mean, logv = self.encode(x)
        z = self.reparametrize(mean, logv)
        return self.decode(z), mean, logv
