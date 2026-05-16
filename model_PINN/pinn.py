from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from model_PINN.constants import *
import matplotlib.pyplot as plt
from tqdm import tqdm


def physics_rhs_tensor(u, constants):
    I = u[:, 0:1]
    Q_col = u[:, 1:2]
    n_col = u[:, 2:3]

    M = Mmaks * (I / 100.0)
    Q_pred = 0.85 * Q * (M / Mmaks) ** (1.0 / 3.0) + (1.0 - 0.85) * Q_col
    n_pred = (M / Mmaks) ** (1.0 / 3.0)
    H = k1 * (nmaks * n_pred) ** 2 + k2 * nmaks * n_pred * Q_pred + k3 * Q_pred ** 2

    denom1 = (nmaks / (n_pred * nmaks))
    inner = 1 - kpd_k_0 - kpd_k_1 * Q_pred * denom1 - kpd_k_2 * Q_pred ** 2 * (denom1 ** 2)
    base = ((n_pred * nmaks) / nmaks) ** 0.36
    kpd = 1 - (inner / (base + 1e-12))
    kpd_adj = er * kpd
    ro = (3600.0 * M * kpd_adj) / (H * g * Q_pred + 1e-12)
    return ro

class PINNNet(nn.Module):
    def __init__(self, in_dim=5, hidden=[128, 128, 64], out_dim=1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def standardize_torch(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(0, keepdim=True)
    if std is None:
        std = x.std(0, unbiased=False, keepdim=True)
        std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def train_pinn_from_two_dfs(df_coll,
                            df_lab,
                            feature_cols,
                            target_col='DENS_LAB',
                            constants=None,
                            epochs=2000,
                            batch_size=128,
                            lambda_data=1.0,
                            lambda_phys=1.0,
                            lr=1e-3,
                            device='cpu',
                            save_prefix='./pinn_run'):
    """
    df_coll : pd.DataFrame
        Коллокации (частая сетка). Должен содержать колонки feature_cols.
    df_lab : pd.DataFrame
        Лабораторный датасет (редкая сетка). Должен содержать те же feature_cols и столбец target_col.
    feature_cols : list[str]
        Список признаков, например ['I','Q','n','x1','x2'].
    target_col : str
        Имя столбца с лабораторным таргетом в df_lab.
    constants : dict
        Константы для physics_rhs_tensor.
    Остальные параметры: параметры обучения, сохранения и т.д.
    """
    if constants is None:
        constants = {}

    X_coll = df_coll[feature_cols].values.astype(np.float32)  # shape [Ncoll, n_features]

    X_lab = df_lab[feature_cols].values.astype(np.float32)
    y_lab = df_lab[target_col].values.astype(np.float32).reshape(-1, 1)

    # -- Преобразуем в torch tensors --
    X_coll_t = torch.tensor(X_coll, dtype=torch.float32, device=device)
    X_lab_t = torch.tensor(X_lab, dtype=torch.float32, device=device)
    y_lab_t = torch.tensor(y_lab, dtype=torch.float32, device=device)

    # -- Нормализация (по collocations) --
    X_coll_std, X_mean, X_stddev = standardize_torch(X_coll_t)
    # сохранение mean/std для последующего инференса
    input_mean = X_mean.detach().cpu().numpy()
    input_std = X_stddev.detach().cpu().numpy()

    # применяем ту же нормализацию к lab-inputs
    X_lab_std = (X_lab_t - X_mean) / (X_stddev + 1e-12)

    # -- Dataloaders --
    coll_dataset = TensorDataset(X_coll_std.detach().cpu(), torch.zeros(X_coll_std.shape[0], 1))
    coll_loader = DataLoader(coll_dataset, batch_size=batch_size, shuffle=False)

    lab_dataset = TensorDataset(X_lab_std.detach().cpu(), y_lab_t.detach().cpu())
    lab_loader = DataLoader(lab_dataset, batch_size=min(batch_size, len(X_lab_std)), shuffle=False)

    # -- Model & optimizer --
    model = PINNNet(in_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'epoch': [], 'loss': [], 'data_loss': [], 'phys_loss': []}

    # -- Training loop --
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss_epoch = 0.0
        data_loss_epoch = 0.0
        phys_loss_epoch = 0.0
        lab_iter = iter(lab_loader)

        for Xc_batch_cpu, _ in coll_loader:
            Xc_batch = Xc_batch_cpu.to(device).float()    # normalized collocation inputs
            rho_pred_coll = model(Xc_batch)

            # unstandardize для вычисления physics_rhs на оригинальной шкале
            Xc_orig = Xc_batch * torch.tensor(input_std, device=device) + torch.tensor(input_mean, device=device)
            rhs_coll = physics_rhs_tensor(Xc_orig, constants).to(device).float()

            L_phys = torch.mean((rho_pred_coll - rhs_coll) ** 2)

            # supervised minibatch (lab)
            try:
                Xl_batch_cpu, yl_batch_cpu = next(lab_iter)
            except StopIteration:
                lab_iter = iter(lab_loader)
                Xl_batch_cpu, yl_batch_cpu = next(lab_iter)

            Xl_batch = Xl_batch_cpu.to(device).float()
            yl_batch = yl_batch_cpu.to(device).float()

            rho_pred_lab = model(Xl_batch)
            L_data = torch.mean((rho_pred_lab - yl_batch) ** 2)

            loss = lambda_data * L_data + lambda_phys * L_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item() * Xc_batch.shape[0]
            data_loss_epoch += L_data.item() * Xc_batch.shape[0]
            phys_loss_epoch += L_phys.item() * Xc_batch.shape[0]

        # усреднение по числу коллокаций (чтобы метрики были сопоставимы с прошлой версией)
        Ncoll = X_coll_std.shape[0]
        total_loss_epoch /= Ncoll
        data_loss_epoch /= Ncoll
        phys_loss_epoch /= Ncoll

        history['epoch'].append(epoch)
        history['loss'].append(total_loss_epoch)
        history['data_loss'].append(data_loss_epoch)
        history['phys_loss'].append(phys_loss_epoch)

        if epoch <= 5 or epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs} | total={total_loss_epoch:.3e} | data={data_loss_epoch:.3e} | phys={phys_loss_epoch:.3e}")

    # -- Сохранение артефактов --
    os.makedirs(os.path.dirname(save_prefix) or '.', exist_ok=True)
    errors_path = save_prefix + "_errors.npy"
    np.save(errors_path, np.vstack([history['loss'], history['data_loss'], history['phys_loss']]))
    model_path = save_prefix + "_model.pth"
    torch.save(model.state_dict(), model_path)

    # -- Графики (каждая метрика на отдельном графике) --
    plt.figure(); plt.semilogy(history['epoch'], history['loss']); plt.title("Total loss (log)"); plt.xlabel("Epoch"); plt.grid(True); plt.savefig(save_prefix + "_total_loss.png"); plt.show()
    plt.figure(); plt.semilogy(history['epoch'], history['data_loss']); plt.title("Data loss (log)"); plt.xlabel("Epoch"); plt.grid(True); plt.savefig(save_prefix + "_data_loss.png"); plt.show()
    plt.figure(); plt.semilogy(history['epoch'], history['phys_loss']); plt.title("Physics loss (log)"); plt.xlabel("Epoch"); plt.grid(True); plt.savefig(save_prefix + "_phys_loss.png"); plt.show()

    return {
        'model': model,
        'input_mean': input_mean,
        'input_std': input_std,
        'history': history,
        'errors_path': errors_path,
        'model_path': model_path
    }

def predict_and_add_column(df_lab: pd.DataFrame,
                           feature_cols: list,
                           model: Optional[torch.nn.Module] = None,
                           input_mean: Optional[np.ndarray] = None,
                           input_std: Optional[np.ndarray] = None,
                           model_path: Optional[str] = None,
                           device: str = 'cpu',
                           batch_size: int = 1024,
                           output_col: str = 'rho_pred_model') -> pd.DataFrame:
    """
    Добавляет столбец output_col в df_lab с предсказаниями PINN.

    Параметры:
    - df_lab: pd.DataFrame с признаками feature_cols
    - feature_cols: список имён признаков в том порядке, в котором модель обучалась
    - model: объект модели PINNNet (если None и указан model_path, модель будет загружена)
    - input_mean: numpy array (1, n_features) или shape (n_features,) — mean, использованный при стандартизации
    - input_std: numpy array — std, использованный при стандартизации
    - model_path: если model is None, можно передать путь к файлу .pth с state_dict
    - device: 'cpu' или 'cuda'
    - batch_size: размер батча для предсказаний
    - output_col: имя столбца для добавляемых предсказаний

    Возвращает тот же df_lab (копию) с добавленным столбцом предсказаний.
    """
    # проверка входов
    if not set(feature_cols).issubset(df_lab.columns):
        missing = set(feature_cols) - set(df_lab.columns)
        raise ValueError(f"В df_lab отсутствуют признаки: {missing}")

    if model is None and model_path is None:
        raise ValueError("Нужно передать либо обученную модель (model), либо путь model_path к весам.")

    # создаём модель если надо
    if model is None:
        # создайте модель с теми же параметрами архитектуры, что и при обучении
        model = PINNNet(in_dim=len(feature_cols))
        map_location = torch.device(device)
        state = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    # проверяем mean/std
    if input_mean is None or input_std is None:
        input_mean = np.array([[50.53506, 93.821465, 69.85917, 0.38352644, 27.461937, 16.402618]])
        input_std = np.array([[10.04325, 44.21287, 15.314799,  0.08437789, 7.308728, 20.35759]])
        # raise ValueError("Нужны input_mean и input_std для нормализации входа.")

    # приводит mean/std к вектору формы (n_features,)
    input_mean = np.asarray(input_mean).reshape(-1)
    input_std = np.asarray(input_std).reshape(-1)
    if input_mean.shape[0] != len(feature_cols) or input_std.shape[0] != len(feature_cols):
        raise ValueError("input_mean/input_std должны соответствовать числу feature_cols")

    # Забираем признаки из df_lab (в порядке feature_cols)
    X = df_lab[feature_cols].values.astype(np.float32)  # shape (N, n_features)

    # Если в df_lab есть NaN в признаках — либо падать, либо пропускать строки.
    if np.isnan(X).any():
        # тут мы заменим NaN на 0 и затем пометим предсказание NaN для этих строк:
        nan_mask = np.any(np.isnan(X), axis=1)
    else:
        nan_mask = np.zeros(X.shape[0], dtype=bool)

    # заменяем NaN временно на 0 для вычислений (предсказания затем занулены)
    X_clean = np.nan_to_num(X, nan=0.0)

    # стандартизация: (X - mean)/std
    X_std = (X_clean - input_mean) / (input_std + 1e-12)

    # предсказания батчами
    preds = np.zeros((X_std.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, X_std.shape[0], batch_size):
            end = start + batch_size
            xb = torch.tensor(X_std[start:end], dtype=torch.float32, device=device)
            yb = model(xb)  # shape [B,1]
            yb_np = yb.squeeze(-1).cpu().numpy()
            preds[start:end] = yb_np

    # Для строк с NaN в входе можно поставить NaN в предсказании:
    preds[nan_mask] = np.nan

    # создаём новый DF (копия) и добавляем столбец
    df_out = df_lab.copy()
    df_out[output_col] = preds

    return df_out