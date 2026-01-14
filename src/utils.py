"""
工具函数模块
提供模型保存/加载、数据处理等辅助功能
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path


def get_model_size(model):
    """
    计算模型大小（MB）
    
    Args:
        model: PyTorch模型
        
    Returns:
        float: 模型大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model):
    """
    统计模型参数量（考虑剪枝mask）
    
    Args:
        model: PyTorch模型
        
    Returns:
        dict: 总参数量、可训练参数量、非零参数量
    """
    total_params = 0
    trainable_params = 0
    non_zero_params = 0
    
    # 遍历所有模块
    for name, module in model.named_modules():
        # 检查weight参数
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight
            # 统计总参数
            total_params += weight.numel()
            # 统计可训练参数（检查weight_orig如果存在，否则检查weight）
            if hasattr(module, 'weight_orig'):
                if module.weight_orig.requires_grad:
                    trainable_params += weight.numel()
            elif weight.requires_grad:
                trainable_params += weight.numel()
            # 统计非零参数（weight会自动应用mask）
            non_zero_params += torch.count_nonzero(weight).item()
        
        # 检查bias参数
        if hasattr(module, 'bias') and module.bias is not None:
            bias = module.bias
            total_params += bias.numel()
            if bias.requires_grad:
                trainable_params += bias.numel()
            non_zero_params += torch.count_nonzero(bias).item()
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_zero': non_zero_params,
        'sparsity': 1 - (non_zero_params / total_params) if total_params > 0 else 0
    }


def measure_inference_time(model, tokenizer, texts, device=None, num_runs=100):
    """
    测量模型推理时间
    
    Args:
        model: PyTorch模型
        tokenizer: 分词器
        texts: 测试文本列表
        device: 设备 (None表示自动选择)
        num_runs: 运行次数
        
    Returns:
        dict: 平均推理时间、吞吐量等指标
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    model.to(device)
    
    # 预热
    with torch.no_grad():
        for text in texts[:5]:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            text = texts[_ % len(texts)]
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    return {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'throughput': 1.0 / np.mean(times)
    }


def save_metrics(metrics, save_path):
    """
    保存评估指标到JSON文件
    
    Args:
        metrics: 指标字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"指标已保存到: {save_path}")


def load_metrics(load_path):
    """
    从JSON文件加载指标
    
    Args:
        load_path: 文件路径
        
    Returns:
        dict: 指标字典
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


def ensure_dir(directory):
    """
    确保目录存在
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def print_model_info(model, model_name="Model"):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    params = count_parameters(model)
    size = get_model_size(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name} 信息:")
    print(f"{'='*50}")
    print(f"总参数量: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")
    print(f"非零参数: {params['non_zero']:,}")
    print(f"稀疏度: {params['sparsity']:.2%}")
    print(f"模型大小: {size:.2f} MB")
    print(f"{'='*50}\n")


class AverageMeter:
    """
    平均值计算器，用于跟踪训练指标
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
