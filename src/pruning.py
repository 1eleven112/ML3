"""
剪枝模块
实现BERT模型的结构化剪枝（注意力头剪枝）和非结构化剪枝
"""

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm


class BERTPruner:
    """
    BERT模型剪枝器
    支持注意力头剪枝、FFN剪枝和全局非结构化剪枝
    """
    
    def __init__(self, model):
        """
        初始化剪枝器
        
        Args:
            model: BERT模型
        """
        self.model = model
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_importance = None
    
    def compute_head_importance(self, dataloader, device='cuda', num_batches=50):
        """
        计算注意力头的重要性分数
        基于梯度和激活值的组合评估
        
        Args:
            dataloader: 数据加载器
            device: 设备
            num_batches: 使用的批次数
            
        Returns:
            importance: 形状为 (num_layers, num_heads) 的重要性矩阵
        """
        print("计算注意力头重要性...")
        
        self.model.to(device)
        self.model.eval()
        
        # 初始化重要性分数
        head_importance = torch.zeros(self.num_layers, self.num_heads).to(device)
        head_mask = torch.ones(self.num_layers, self.num_heads).to(device)
        head_mask.requires_grad_(True)
        
        # 计算重要性
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="计算重要性", total=min(num_batches, len(dataloader)))):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                head_mask=head_mask
            )
            
            loss = outputs.loss
            loss.backward()
            
            # 累积重要性（梯度的绝对值）
            head_importance += head_mask.grad.abs().detach()
            
            total_loss += loss.item()
            num_samples += input_ids.size(0)
            
            # 清空梯度
            head_mask.grad = None
            self.model.zero_grad()
        
        # 归一化重要性分数
        head_importance = head_importance / num_batches
        
        self.head_importance = head_importance.cpu()
        
        print(f"重要性计算完成！平均损失: {total_loss / num_batches:.4f}")
        
        return self.head_importance
    
    def get_heads_to_prune(self, prune_ratio=0.5):
        """
        根据重要性分数选择要剪枝的注意力头
        
        Args:
            prune_ratio: 剪枝比例（0-1之间）
            
        Returns:
            heads_to_prune: 字典 {layer: [head_indices]}
        """
        if self.head_importance is None:
            raise ValueError("请先调用 compute_head_importance() 计算重要性")
        
        # 展平重要性分数
        importance_flat = self.head_importance.view(-1)
        num_total_heads = len(importance_flat)
        num_to_prune = int(num_total_heads * prune_ratio)
        
        # 找出重要性最低的头
        _, indices = torch.topk(importance_flat, num_to_prune, largest=False)
        
        # 转换为 {layer: [heads]} 格式
        heads_to_prune = {}
        for idx in indices:
            layer = idx.item() // self.num_heads
            head = idx.item() % self.num_heads
            
            if layer not in heads_to_prune:
                heads_to_prune[layer] = []
            heads_to_prune[layer].append(head)
        
        # 打印剪枝信息
        print(f"\n将剪枝 {num_to_prune}/{num_total_heads} 个注意力头 ({prune_ratio:.1%})")
        for layer, heads in sorted(heads_to_prune.items()):
            print(f"  Layer {layer}: {len(heads)} 个头 - {heads}")
        
        return heads_to_prune
    
    def prune_attention_heads(self, heads_to_prune):
        """
        执行注意力头剪枝
        
        Args:
            heads_to_prune: 字典 {layer: [head_indices]}
        """
        print("\n执行注意力头剪枝...")
        
        # 使用transformers库的剪枝功能
        from transformers.models.bert.modeling_bert import prune_linear_layer
        
        for layer in range(self.num_layers):
            if layer not in heads_to_prune or len(heads_to_prune[layer]) == 0:
                continue
            
            # 获取该层的attention模块
            attention = self.model.bert.encoder.layer[layer].attention.self
            
            # 要保留的头索引
            heads_to_keep = [h for h in range(self.num_heads) if h not in heads_to_prune[layer]]
            
            # 简化实现：将不重要的注意力头的权重置零
            head_size = self.config.hidden_size // self.num_heads
            for head_idx in heads_to_prune[layer]:
                start_idx = head_idx * head_size
                end_idx = (head_idx + 1) * head_size
                
                # 将query、key、value的权重置零
                with torch.no_grad():
                    attention.query.weight[start_idx:end_idx, :] = 0
                    attention.key.weight[start_idx:end_idx, :] = 0
                    attention.value.weight[start_idx:end_idx, :] = 0
                    
                    if attention.query.bias is not None:
                        attention.query.bias[start_idx:end_idx] = 0
                        attention.key.bias[start_idx:end_idx] = 0
                        attention.value.bias[start_idx:end_idx] = 0
        
        print("注意力头剪枝完成！")
    
    def apply_unstructured_pruning(self, sparsity=0.4):
        """
        应用非结构化（全局）剪枝
        根据权重大小剪枝，不考虑结构
        
        Args:
            sparsity: 稀疏度（0-1之间）
        """
        print(f"\n应用非结构化剪枝（稀疏度: {sparsity:.1%}）...")
        
        # 收集所有要剪枝的参数
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        # 应用全局非结构化剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        # 永久化剪枝（移除mask，直接修改权重）
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        print("非结构化剪枝完成！")
    
    def apply_magnitude_pruning(self, sparsity_per_layer=0.5):
        """
        对每层应用基于权重大小的剪枝
        
        Args:
            sparsity_per_layer: 每层的稀疏度
        """
        print(f"\n应用magnitude剪枝（每层稀疏度: {sparsity_per_layer:.1%}）...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity_per_layer)
                prune.remove(module, 'weight')
        
        print("Magnitude剪枝完成！")


class AdaptivePruner:
    """
    自适应剪枝器：不同层使用不同的剪枝率
    这是本项目的创新点之一
    """
    
    def __init__(self, model):
        """
        初始化自适应剪枝器
        
        Args:
            model: BERT模型
        """
        self.model = model
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
    
    def apply_layer_adaptive_pruning(self, base_sparsity=0.5):
        """
        应用层级自适应剪枝
        浅层保留更多参数，深层更激进剪枝
        
        Args:
            base_sparsity: 基础稀疏度
        """
        print(f"\n应用层级自适应剪枝（基础稀疏度: {base_sparsity:.1%}）...")
        
        for layer_idx in range(self.num_layers):
            # 计算该层的稀疏度：深层更高
            # 浅层: 0.3, 中层: 0.5, 深层: 0.7
            layer_sparsity = base_sparsity * (1 + 0.4 * (layer_idx / self.num_layers))
            layer_sparsity = min(layer_sparsity, 0.8)  # 最大80%稀疏度
            
            # 获取该层的所有Linear模块
            encoder_layer = self.model.bert.encoder.layer[layer_idx]
            
            for name, module in encoder_layer.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=layer_sparsity)
                    prune.remove(module, 'weight')
            
            print(f"  Layer {layer_idx}: 稀疏度 {layer_sparsity:.1%}")
        
        print("层级自适应剪枝完成！")


def prune_bert_model(model, method='heads', prune_ratio=0.5, dataloader=None, device='cuda'):
    """
    便捷函数：剪枝BERT模型
    
    Args:
        model: BERT模型
        method: 剪枝方法 ('heads', 'unstructured', 'adaptive')
        prune_ratio: 剪枝比例
        dataloader: 数据加载器（头剪枝需要）
        device: 设备
        
    Returns:
        pruned_model: 剪枝后的模型
    """
    pruner = BERTPruner(model)
    
    if method == 'heads':
        if dataloader is None:
            raise ValueError("注意力头剪枝需要提供dataloader")
        
        # 计算重要性并剪枝
        pruner.compute_head_importance(dataloader, device=device)
        heads_to_prune = pruner.get_heads_to_prune(prune_ratio)
        pruner.prune_attention_heads(heads_to_prune)
        
    elif method == 'unstructured':
        pruner.apply_unstructured_pruning(sparsity=prune_ratio)
        
    elif method == 'adaptive':
        adaptive_pruner = AdaptivePruner(model)
        adaptive_pruner.apply_layer_adaptive_pruning(base_sparsity=prune_ratio)
        
    else:
        raise ValueError(f"不支持的剪枝方法: {method}")
    
    return model
