import numpy as np
import torch
import torch.nn as nn
import os
import json
import xarray as xr
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 模型架构定义
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_len * n_features, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())    
        # 注意：此处自回归单步推进默认 output_len=1
        layers.append(nn.Linear(hidden_dim, output_len * n_features))
        self.fc = nn.Sequential(*layers)
        self.output_len = output_len
        self.n_features = n_features

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y.view(-1, self.output_len, self.n_features)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute 1D Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Initialize output tensor in frequency domain
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply the lower Fourier modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space via inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

# added Unet
class DoubleConv1d(nn.Module):
    """(Conv1d => BatchNorm1d => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down1d(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up1d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1D 反卷积进行上采样
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理输入特征长度(n_features)不能被2整除导致的大小不匹配问题
        diff = x2.size(-1) - x1.size(-1)
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        
        # 沿着 channel 维度进行拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet1d(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim=64, n_layers=3):
        """
        参数映射:
        - input_len: 输入的时间步 (对应 U-Net 的 in_channels)
        - output_len: 预测的时间步 (对应 U-Net 的 out_channels)
        - n_features: 空间网格数 (卷积沿此维度滑动，为兼容性保留此参数)
        - hidden_dim: 初始基础通道数 (控制模型宽度)
        - n_layers: 下采样/上采样的层数 (控制模型深度)
        """
        super(UNet1d, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.n_layers = max(1, n_layers)  # 至少需要 1 层 U-Net 结构

        # 初始特征提取
        self.inc = DoubleConv1d(input_len, hidden_dim)

        # 动态构建下采样路径 (Encoder)
        self.downs = nn.ModuleList()
        for i in range(self.n_layers):
            in_c = hidden_dim * (2 ** i)
            out_c = hidden_dim * (2 ** (i + 1))
            self.downs.append(Down1d(in_c, out_c))

        # 动态构建上采样路径 (Decoder)
        self.ups = nn.ModuleList()
        for i in reversed(range(self.n_layers)):
            in_c = hidden_dim * (2 ** (i + 1))
            out_c = hidden_dim * (2 ** i)
            self.ups.append(Up1d(in_c, out_c))

        # 最终输出层，映射回所需的输出时间步长度
        self.outc = nn.Conv1d(hidden_dim, output_len, kernel_size=1)

    def forward(self, x):
        # 此时 x 的 shape 为 [batch, input_len, n_features]
        # 在 1D 卷积中，通道维度天然是 input_len，序列长度天然是 n_features，因此无需 permute 转换维度
        
        x = self.inc(x)
        skips = [x]

        # Encoder 前向传播
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # 弹出最底层的特征（Bottleneck），不参与 skip connection 拼接
        skips.pop()

        # Decoder 前向传播
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        logits = self.outc(x)
        
        # 输出 shape 为 [batch, output_len, n_features]，与原先 DataLoader 目标一致
        return logits

# ==========================================
# 2. 数据处理辅助函数 (针对 n 步预测修改)
# ==========================================
def generate_ar_sequences(data, input_len=3, n_steps=10):
    """
    针对自回归测试生成序列。
    Y 的长度不再是 1，而是评估所需的完整未来 n_steps，作为 Ground Truth。
    """
    X = []
    Y = []
    for i in range(len(data) - input_len - n_steps + 1):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + n_steps])
    X = np.array(X)
    Y = np.array(Y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

class SimpleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. 核心功能：自回归预测与 Xarray 封装
# ==========================================
def load_trained_model(model_path, hparams, device):
    """实例化模型并加载预训练权重"""

    if hparams.get('model_type', 'MLP') == 'FNO':
        model = FNO1d(
            input_len=hparams['input_len'], 
            output_len=hparams['output_len'], 
            n_features=hparams['n_features'], 
            hidden_dim=hparams['hidden_dim'], 
            n_layers=hparams['n_layers'],
        )
    elif hparams.get('model_type') == 'UNet':
        model = UNet1d(
            input_len=hparams['input_len'], 
            output_len=hparams['output_len'], 
            n_features=hparams['n_features'], 
            hidden_dim=hparams['hidden_dim'], 
            n_layers=hparams['n_layers']
        )

    else:
        model = MLP(
            input_len=hparams['input_len'], 
            output_len=hparams['output_len'], 
            n_features=hparams['n_features'], 
            hidden_dim=hparams['hidden_dim'], 
            n_layers=hparams['n_layers']
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model_ar(model, test_loader, device, n_steps):
    """在测试集上执行 n 步自回归预测"""
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)  # shape: (B, n_steps, n_features)
            
            curr_X = batch_X
            batch_preds = []
            
            # 自回归滚动预测循环
            for step in range(n_steps):
                pred = model(curr_X)  # shape: (B, 1, n_features)
                batch_preds.append(pred)
                
                # 滚动更新输入：丢弃最老的时间步，拼接最新的预测结果
                curr_X = torch.cat([curr_X[:, 1:, :], pred], dim=1)
                
            # 将 list 中的张量拼接为 (B, n_steps, n_features)
            batch_preds = torch.cat(batch_preds, dim=1)
            
            # 计算全局指标
            total_mse += criterion_mse(batch_preds, batch_Y).item() * batch_X.size(0)
            total_mae += criterion_mae(batch_preds, batch_Y).item() * batch_X.size(0)
            
            all_preds.append(batch_preds.cpu().numpy())
            all_trues.append(batch_Y.cpu().numpy())

    avg_mse = total_mse / len(test_loader.dataset)
    avg_mae = total_mae / len(test_loader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    print(f"AR ({n_steps}-step) Test Results | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f}")
    return avg_mse, avg_mae, all_preds, all_trues

def save_to_xarray(preds, trues, mse, mae, save_path):
    """将预测结果、真实值和度量指标保存为 Xarray Dataset"""
    n_samples, n_steps, n_features = preds.shape
    
    ds = xr.Dataset(
        data_vars={
            "predictions": (["sample", "time_step", "feature"], preds),
            "ground_truth": (["sample", "time_step", "feature"], trues),
            # 可选：保存绝对误差场
            "absolute_error": (["sample", "time_step", "feature"], np.abs(preds - trues))
        },
        coords={
            "sample": np.arange(n_samples),
            "time_step": np.arange(1, n_steps + 1),  # 表示未来第 1 到 n 步
            "feature": np.arange(n_features)
        },
        attrs={
            "description": "Autoregressive Prediction Results",
            "test_MSE": float(mse),
            "test_MAE": float(mae),
            "n_steps_ahead": n_steps
        }
    )
    
    ds.to_netcdf(save_path)
    print(f"Results successfully saved to {save_path}")

# ==========================================
# 主执行入口
# ==========================================
if __name__ == '__main__':
    parser = ArgumentParser(description="Autoregressive Testing for Pretrained Model")
    parser.add_argument('--model_path', type=str, default='/workspace/git/DGNN/DIS_stochastic/weight_analysis/run_20260423_035911_Standard/last_model.pth', help="Path to the pretrained model weights")
    parser.add_argument('--test_data_path', type=str, default='/workspace/git/DGNN/data/ks/test/data.npy', help="Path to the test dataset (numpy file)")
    parser.add_argument('--ar_steps', type=int, default=20, help="Number of autoregressive steps to predict")
    args = parser.parse_args()

    # 1. 设定路径与自回归步数
    model_path = args.model_path
    test_data_path = args.test_data_path
    
    # 设定自回归预测未来的步数 n
    AR_STEPS = args.ar_steps
    
    model_dir = os.path.dirname(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. 读取超参数
    hparams_path = os.path.join(model_dir, 'hparams.json')
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Hyperparameter file not found at '{hparams_path}'.")
        
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # 确保基础模型是单步预测模型 (output_len = 1)，这是自回归的基础
    assert hparams['output_len'] == 1, "Autoregressive rollout requires the base model to output 1 step at a time."
    
    # 3. 准备自回归测试数据
    test_data = np.load(test_data_path)
    # flatten space dim
    test_data = test_data.reshape(test_data.shape[0], -1)  # shape: (T, n_features)
    X_test_tensor, Y_test_tensor = generate_ar_sequences(test_data, hparams['input_len'], n_steps=AR_STEPS)
    test_dataset = SimpleDataset(X_test_tensor, Y_test_tensor)
    
    batch_size = hparams.get('batch_size', 128)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. 加载模型
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, hparams, device)
    
    # 5. 执行自回归测试
    print(f"Executing {AR_STEPS}-step autoregressive evaluation...")
    mse, mae, all_preds, all_trues = test_model_ar(model, test_loader, device, n_steps=AR_STEPS)
    
    # 6. 打包保存到 xarray NetCDF 文件
    xr_save_path = os.path.join(model_dir, 'ar_test_results.nc')
    save_to_xarray(all_preds, all_trues, mse, mae, xr_save_path)