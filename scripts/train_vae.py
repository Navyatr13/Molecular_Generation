import torch
import time
import torch.nn.functional as F
from tqdm import tqdm

def loss_function(recon_out, data, mean, logv):

    """Loss function combining reconstruction loss and KL divergence."""
    BCE = F.binary_cross_entropy(recon_out, data, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    return BCE + KLD

def validate_vae(model, val_loader, device):
    """Run validation on the VAE model."""
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for data in val_loader:
            data = data[0].transpose(1, 2).to(device)
            recon_batch, mean, logvar = model(data)
            loss = loss_function(recon_batch, data.transpose(1, 2), mean, logvar)
            val_loss += loss.item()
    return val_loss / len(val_loader.dataset)

def train_vae(model, train_loader, val_loader, optimizer, device, epochs):
    """Train the VAE model and validate after each epoch."""
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, data in enumerate(tepoch):
                data = data[0].transpose(1, 2).to(device)
                optimizer.zero_grad()
                recon_batch, mean, logvar = model(data)
                loss = loss_function(recon_batch, data.transpose(1, 2), mean, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
    
            # Calculate the average training loss
            train_loss = train_loss / len(train_loader.dataset)
            
            # Run validation and calculate validation loss
            val_loss = validate_vae(model, val_loader, device)
            
            epoch_time = time.time() - start_time
            print(f"End of Epoch [{epoch+1}]: Train Loss = {train_loss:.2f} Validation Loss = {val_loss:.2f} Time: {epoch_time:.2f}s")


    return model
    
def fine_tune_vae(model, train_loader, val_loader, optimizer, device, config):
    """Fine-tune the VAE model with new data (EGFR)."""
    model.train()
    epochs = config['training']['fine_tune_epochs']
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, data in enumerate(tepoch):
                data = data[0].transpose(1, 2).to(device)
                optimizer.zero_grad()
                recon_batch, mean, logvar = model(data)
                loss = loss_function(recon_batch, data.transpose(1, 2), mean, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # Validation step (if desired)
        val_loss = evaluate_vae(model, val_loader, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

    # Save the fine-tuned model after training
    torch.save(model.state_dict(), config['training']['fine_tuned_model_path'])
    return model

def evaluate_vae(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data[0].transpose(1, 2).to(device)
            recon_batch, mean, logvar = model(data)
            val_loss += loss_function(recon_batch, data.transpose(1, 2), mean, logvar).item()
    return val_loss / len(val_loader.dataset)
