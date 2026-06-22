# original work only on single variable, we use L2 norm energy for comparison
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import json
import datetime
import argparse
from torch.utils.data import Dataset, DataLoader

# Import the official DenseWeight library
from denseweight import DenseWeight

# ==========================================
# Functions & Modules
# ==========================================

class DenseLoss(nn.Module):
    """Weighted MSE loss. Computes per-sample MSE, then scales each sample by its
    precomputed weight. A tiny noise term is added to keep gradients non-zero when
    a sample weight is (near) zero."""
    def __init__(self, base_loss=None):
        super(DenseLoss, self).__init__()
        if base_loss is None:
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = base_loss

    def forward(self, y_pred, y_true, sample_weights):
        loss_per_sample = self.base_loss(y_pred, y_true)
        # Align the weight tensor rank with the per-sample loss tensor
        if loss_per_sample.dim() > sample_weights.dim():
            sample_weights = sample_weights.view(-1, 1)
        elif loss_per_sample.dim() < sample_weights.dim():
            sample_weights = sample_weights.squeeze()
            # add a random small noise to avoid zero weights causing zero loss
        weighted_loss = loss_per_sample * sample_weights + 1e-8 * torch.rand_like(sample_weights)
        return torch.mean(weighted_loss)

def generate_sequences(data, input_len=3, output_len=1):
    """Build autoregressive (input -> output) windows from a flat time series.
    Returns X of shape (N, input_len, n_features) and Y of shape (N, output_len, n_features)."""
    X = []
    Y = []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + output_len])
    X = np.array(X)
    Y = np.array(Y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_model(model, train_loader, val_loader, optimizer, criterion, device, save_dir, num_epochs=100, patience=10):
    """Standard train/validate loop with early stopping on validation MSE.
    Saves best/last checkpoints and the loss history to save_dir."""
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    # Validation always uses unweighted MSE so the metric is comparable across methods
    val_criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y, batch_weights in train_loader:
            batch_X, batch_Y, batch_weights = batch_X.to(device), batch_Y.to(device), batch_weights.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            # During the warmup phase, override with uniform weights so the model first
            # learns a stable map before the weighting scheme is engaged
            if epoch < args.warmup:
                # print(f"Epoch {epoch+1}: Warmup phase - using uniform weights")
                batch_weights = torch.ones_like(batch_weights)
            loss = criterion(predictions, batch_Y, batch_weights)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_X, val_Y, _ in val_loader: 
                val_X, val_Y = val_X.to(device), val_Y.to(device)
                val_preds = model(val_X)
                v_loss = val_criterion(val_preds, val_Y)
                val_loss += v_loss.item() * val_X.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1:04d}/{num_epochs:04d}] | Train Loss: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")
        
        # --- Early Stopping & Model Saving ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}!")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
    
    with open(os.path.join(save_dir, 'loss_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining Complete. Best Val MSE: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history

class MLP(nn.Module):
    """Simple fully-connected surrogate. Flattens the (input_len, n_features) window,
    passes it through n_layers hidden layers, and reshapes to (output_len, n_features)."""
    def __init__(self, input_len, output_len, n_features, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_len * n_features, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_len * n_features))
        self.fc = nn.Sequential(*layers)
        self.output_len = output_len
        self.n_features = n_features

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y.view(-1, self.output_len, self.n_features)

def compute_daw_weights(d, alpha):
    """Compute the Dimension-Aware Weighting (DAW) per-sample weights from the local dimension d.

    Pipeline (matches Sec. 2.2 of the paper):
      Step 1-2: inverse-density (KDE) weighting over the distribution of d via the
                official DenseWeight package -> upweights BOTH tails of P(d) symmetrically.
      Step 3:   multiplicative d-tilt (multiply by the min-max-normalized d) -> breaks the
                symmetry, suppressing the rare low-d tail and amplifying the rare high-d tail.
      Step 4:   global mean-normalization so that E[w] = 1.0, preserving the gradient scale.

    Returns a float32 numpy array of the same length as d.
    """
    # Step 1-2: inverse-density (KDE-based) weights over d
    dw = DenseWeight(alpha=alpha)
    initial_weights_np = dw.fit(d)

    # Step 3a: min-max normalize d to [0, 1]
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max > d_min:
        d_normalized = (d - d_min) / (d_max - d_min)
    else:
        # Add a small epsilon tolerance to prevent division-by-zero errors
        d_normalized = np.ones_like(d)

    # Step 3b: apply the d-tilt (suppress low-d, amplify high-d)
    modified_weights_np = initial_weights_np * d_normalized

    # Step 4: mean-normalize so the expected weight is 1.0
    final_weights_np = modified_weights_np / (np.mean(modified_weights_np) + 1e-8)
    return final_weights_np.astype(np.float32)


class ImbalancedDataset(Dataset):
    """Dataset that returns (input, target, per-sample weight) triplets."""
    def __init__(self, X, Y, weights):
        self.X = X
        self.Y = Y
        self.weights = weights

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.weights[idx]

# ==========================================
# Main Execution
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train surrogate model with DAW / DenseWeight / Standard weighting")
    
    # --- Weighting scheme selection ---
    # 'DAW'         : dimension-aware weighting (local dimension d as prior)
    # 'DenseWeight' : target-space density weighting using the spatial L2-norm
    # 'RandomWeight': shuffled-weight ablation (the exact DAW weight set under a random permutation)
    # 'Standard'    : uniform weighting (reference baseline)
    parser.add_argument('--method', type=str, default='DenseWeight', choices=['DenseWeight', 'Standard', 'DAW', 'RandomWeight'], help='Select training mode: DenseWeight, Standard, DAW, or RandomWeight')
    parser.add_argument('--alpha', type=float, default=0.5, help='Intensity parameter controlling how aggressively rare/high-d regions are upweighted')

    # --- Data / model dimensions ---
    parser.add_argument('--n_features', type=int, default=64, help='Number of spatial grid points (features) per state')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer width of the MLP')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of MLP layers')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--input_len', type=int, default=3, help='Number of input time steps fed to the model')
    parser.add_argument('--output_len', type=int, default=1, help='Number of output time steps the model predicts')

    # --- Optimization ---
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-9, help='L2 weight decay for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=900, help='Early-stopping patience (epochs without val improvement)')

    # --- I/O paths (replace with your own local directories) ---
    parser.add_argument('--base_save_dir', type=str, default='/path/to/save_dir/daw/ks', help='Root directory where run logs and checkpoints are written')
    parser.add_argument('--data_dir', type=str, default='/path/to/data_dir/ks', help='Directory containing train/val/test data and precomputed d values')
    parser.add_argument('--d_path', type=str, default=None, help='Path to the precomputed local dimension d .npy file (required for DAW and RandomWeight; defaults to <data_dir>/train/d_sample_pair_in3_out1_q0.99.npy if not set)')

    # --- Reproducibility / device ---
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--random_seed', action='store_true', help='use random seed instead of fixed seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup epochs trained with uniform weights before the weighting scheme is applied')
    
    args = parser.parse_args()
    hparams = vars(args)
    
    # ==========================
    # 1. save path & seeds
    # ==========================
    os.makedirs(args.base_save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.random_seed:
        args.seed = np.random.randint(0, 10000)
        print(f"Using random seed: {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ==========================
    # 2. initialize logging
    # ==========================
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tag the run folder with the method and alpha so different configurations are easy to separate
    mode_str = f"{args.method}_alpha{args.alpha}"
    
    run_name = f"run_{run_timestamp}_{mode_str}"
    run_dir = os.path.join(args.base_save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Persist the full hyperparameter set for this run
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)
        
    print(f"--- Training initialized in {mode_str} mode. Logs saved to: {run_dir} ---")

    # ==========================
    # 3. load data
    # ==========================
    # Precomputed per-sample local dimension d (used by the DAW and RandomWeight methods).
    # Use the explicit --d_path if provided; otherwise fall back to the default location under data_dir.
    d_path = args.d_path if args.d_path is not None else os.path.join(args.data_dir, 'train', 'd_sample_pair_in3_out1_q0.99.npy')
    data_path = os.path.join(args.data_dir, 'train', 'data.npy')
    val_data_path = os.path.join(args.data_dir, 'val', 'data.npy')
    test_data_path = os.path.join(args.data_dir, 'test', 'data.npy')

    data = np.load(data_path)
    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    # flatten the data to 2D (N, n_features)
    data = data.reshape(data.shape[0], -1)
    val_data = val_data.reshape(val_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    # Convert raw time series into supervised (input window -> output window) pairs
    X_train_tensor, Y_train_tensor = generate_sequences(data, args.input_len, args.output_len)
    X_val_tensor, Y_val_tensor = generate_sequences(val_data, args.input_len, args.output_len)
    X_test_tensor, Y_test_tensor = generate_sequences(test_data, args.input_len, args.output_len)

    # ==========================
    # 4. Mode Control: DenseWeight vs DAW vs RandomWeight vs Standard
    # ==========================
    if args.method == 'DenseWeight':
        print("Calculating weights using target's Spatial L2-Norm (DenseWeight baseline)...")
        
        # 1. Extract the target variable numpy array
        # Y_train_tensor has shape (N, output_len, n_features)
        y_numpy = Y_train_tensor.numpy()
        
        # 2. Compute the L2 norm along the spatial/feature dimension (axis=-1) (i.e., sqrt of sum of squares)
        # This reduces the dimensionality, yielding a scalar representing each sample's "spatial energy"
        target_l2_norm = np.linalg.norm(y_numpy, axis=-1).flatten()
        
        # 3. Use the official DenseWeight package to fit this scalar distribution
        dw = DenseWeight(alpha=args.alpha)
        weights_np = dw.fit(target_l2_norm)
        
        # 4. Convert to tensor
        global_weights = torch.tensor(weights_np, dtype=torch.float32)
        print("DenseWeight (L2-Norm) Weight calculation complete.")
        # save weights
        np.save(os.path.join(run_dir, 'denseweight_weights.npy'), weights_np)

    elif args.method == 'DAW':
        print("Calculating weights using DAW method...")
        # 0. Verify the precomputed d file exists before proceeding; DAW cannot run without it
        if not os.path.isfile(d_path):
            raise FileNotFoundError(
                f"DAW method requires the precomputed local dimension file, but it was not found at: {d_path}. "
                f"Provide a valid path via --d_path, or place the file at the default location under --data_dir."
            )
        # 1. Load the precomputed local dimension d values
        d = np.load(d_path).squeeze()

        # 2. Compute the full DAW weights (inverse-density on d, d-tilt, mean-normalization)
        final_weights_np = compute_daw_weights(d, args.alpha)

        # 3. Save the final reshaped weights
        np.save(os.path.join(run_dir, 'daw_weights.npy'), final_weights_np)

        # 4. Convert to tensor
        global_weights = torch.tensor(final_weights_np, dtype=torch.float32)
        print("DAW Weight calculation (with target distribution reshaping) complete.")

    elif args.method == 'RandomWeight':
        # RandomWeight ablation: take the EXACT DAW weight set, then destroy its alignment
        # with high-d states via a uniformly random permutation. This isolates whether DAW's
        # gain stems from where the weight is placed (alignment with d) rather than merely from
        # the increased gradient variance that any non-uniform weighting induces.
        print("Calculating weights using RandomWeight (shuffled-DAW ablation)...")
        # 0. Verify the precomputed d file exists; RandomWeight reuses the DAW weights
        if not os.path.isfile(d_path):
            raise FileNotFoundError(
                f"RandomWeight method requires the precomputed local dimension file, but it was not found at: {d_path}. "
                f"Provide a valid path via --d_path, or place the file at the default location under --data_dir."
            )
        # 1. Load d and compute the same DAW weights used by the DAW method
        d = np.load(d_path).squeeze()
        daw_weights = torch.tensor(compute_daw_weights(d, args.alpha), dtype=torch.float32)

        # 2. Truncate to align with the number of training samples (no-op if lengths already match)
        num_samples = len(Y_train_tensor)
        daw_weights = daw_weights[:num_samples]

        # 3. Shuffle the DAW weights with a seeded permutation for reproducibility.
        #    The weight-magnitude distribution is preserved, but the weight<->high-d correspondence is broken.
        shuffled_indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(args.seed))
        global_weights = daw_weights[shuffled_indices]

        # 4. Save the shuffled weights for reproducibility
        np.save(os.path.join(run_dir, 'randomweight_weights.npy'), global_weights.numpy())
        print("=> Applied Shuffled DAW weights (RandomWeight ablation).")

    else:
        # Standard mode: equal weights
        global_weights = torch.ones(len(Y_train_tensor), dtype=torch.float32)

    # Wrap tensors into datasets; validation and test always use uniform weights
    train_dataset = ImbalancedDataset(X_train_tensor, Y_train_tensor, global_weights)
    val_dataset = ImbalancedDataset(X_val_tensor, Y_val_tensor, torch.ones(len(Y_val_tensor)))
    test_dataset = ImbalancedDataset(X_test_tensor, Y_test_tensor, torch.ones(len(Y_test_tensor)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================
    # 5. model, criterion, optimizer
    # ==========================
    model = MLP(args.input_len, args.output_len, args.n_features, 
                args.hidden_dim, args.n_layers).to(device)
                
    criterion = DenseLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trained_model, loss_history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device,
        save_dir=run_dir,
        num_epochs=args.num_epochs, 
        patience=args.patience
    )