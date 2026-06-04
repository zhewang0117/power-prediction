#!/usr/bin/env python3
"""
生成 VMD-BiLSTM 预测, 格式与 GWO-ELM 校正器兼容.
输出: data/predictions/vmd_bilstm_pred_{year}.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch

from train_vmd_bilstm import Config, build_sequences, BiLSTMRegressor, set_seed

def generate_predictions(year: int, model_path: str, out_path: str):
    cfg = Config()
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    # 加载数据
    fp = f'data/power/ghdt_merged_15min_{year}.csv'
    if not os.path.exists(fp):
        print(f"ERROR: {fp} not found")
        return

    df = pd.read_csv(fp)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'VALUE', 'wind_speed_10m (m/s)', 'wind_direction_10m (°)']]
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    print(f"Loaded {year}: {len(df)} rows")

    # 构建序列
    print("Building sequences (VMD)...")
    t0 = time.time()
    x_past, x_fut, y_true, t_start, _ = build_sequences(
        df, cfg, use_vmd=True, use_typhoon=False)
    print(f"  Samples: {len(y_true)}, past={x_past.shape}, fut={x_fut.shape} ({time.time()-t0:.0f}s)")

    # 加载模型
    input_size = x_past.shape[2]
    exog_size = x_fut.shape[2]
    model = BiLSTMRegressor(input_size, exog_size, cfg.hidden_size,
                             cfg.num_layers, cfg.dropout)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Model loaded: input={input_size}, exog={exog_size}")

    # 推理
    print("Running inference...")
    t0 = time.time()
    all_preds = []
    batch_size = cfg.batch_size
    with torch.no_grad():
        for i in range(0, len(y_true), batch_size):
            xb = torch.from_numpy(x_past[i:i+batch_size]).float().to(device)
            xf = torch.from_numpy(x_fut[i:i+batch_size]).float().to(device)
            pred = model(xb, xf).cpu().numpy()
            all_preds.append(pred)
    pred_norm = np.concatenate(all_preds, axis=0)
    print(f"  Inference done: {pred_norm.shape} ({time.time()-t0:.0f}s)")

    # 还原到 MW
    pred_mw = pred_norm * cfg.capacity
    true_mw = y_true * cfg.capacity

    # 保存 (与 bilstm_pred_*.csv 同格式)
    h = cfg.horizon
    out = {"target_start_time": t_start}
    for i in range(h):
        out[f"y_true_t+{i+1}_mw"] = true_mw[:, i]
    for i in range(h):
        out[f"y_pred_t+{i+1}_mw"] = pred_mw[:, i]

    pred_df = pd.DataFrame(out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(pred_df)} rows, {len(pred_df.columns)} cols)")
    print("Done!")

if __name__ == '__main__':
    year = 2020
    model_path = "vmd_bilstm_best.pt"
    out_path = f"data/predictions/vmd_bilstm_pred_{year}.csv"
    generate_predictions(year, model_path, out_path)
