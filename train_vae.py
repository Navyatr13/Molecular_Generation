import torch
import torch.nn.functional as F

def loss_function(recon_out, data, mean, logv):
    """Loss function combining reconstruction loss and KL divergence."""
    BCE = F.binary_cross_entropy(recon_out, data, size_average=False)
    KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    return BCE + KLD

def train_vae(model, train_loader, optimizer, device, epochs):
    """Train the VAE model."""
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data[0].transpose(1, 2).to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar = model(data)
            loss = loss_function(recon_batch, data.transpose(1, 2), mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} / Batch {batch_idx}: Loss {loss:.4f}')

        print(f'End of Epoch {epoch}: Train Loss {train_loss / len(train_loader.dataset):.4f}')

    return model
