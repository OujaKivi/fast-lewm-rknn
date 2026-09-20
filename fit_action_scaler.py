"""
从 pusht_expert_train 数据集中拟合 action StandardScaler
"""
import sys
sys.path.insert(0, '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/Fast-LeWorldModel')

import os
import numpy as np
import stable_worldmodel as swm
from sklearn import preprocessing
import json

DATASET_PATH = '/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/data/datasets/datasets--galilai-group--lewm-pusht/snapshots/ea321e392348e3c65a18ab0d685f00e57be2c3e0/pusht_expert_train.lance'

print('=== 加载数据集 ===')
print(f'路径: {DATASET_PATH}')
print(f'存在: {os.path.exists(DATASET_PATH)}')

# 尝试用 LanceDataset 加载
try:
    dataset = swm.data.LanceDataset(
        'pusht_expert_train',
        keys_to_cache=['action', 'proprio', 'state'],
        cache_dir=os.path.dirname(DATASET_PATH),
    )
    print('LanceDataset 加载成功')
    print(f'列名: {dataset.column_names}')
    print(f'长度: {len(dataset)}')
    
    # 获取 action 数据
    action_data = dataset.get_col_data('action')
    print(f'\naction 数据 shape: {action_data.shape}')
    print(f'action 数据 (前5行): {action_data[:5]}')
    
    # 去掉 NaN
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    print(f'去掉 NaN 后 shape: {action_data.shape}')
    
    # 拟合 StandardScaler
    scaler = preprocessing.StandardScaler()
    scaler.fit(action_data)
    
    print(f'\n=== StandardScaler 参数 ===')
    print(f'mean: {scaler.mean_}')
    print(f'std (scale_): {scaler.scale_}')
    print(f'var: {scaler.var_}')
    
    # 保存参数
    params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'var': scaler.var_.tolist(),
        'n_samples': len(action_data),
    }
    with open('/Users/wangjiwei/Doubao/chats/2026-09-14/new-chat-2/action_scaler_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f'\n参数已保存到 action_scaler_params.json')
    
    # 同时拟合 proprio 和 state
    for col in ['proprio', 'state']:
        if col in dataset.column_names:
            col_data = dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            col_scaler = preprocessing.StandardScaler()
            col_scaler.fit(col_data)
            print(f'\n{col} mean: {col_scaler.mean_}')
            print(f'{col} std: {col_scaler.scale_}')
    
except Exception as e:
    print(f'LanceDataset 加载失败: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
