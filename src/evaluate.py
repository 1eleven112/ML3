"""
评估模块
提供模型性能评估、指标计算和结果可视化功能
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class ModelEvaluator:
    """
    模型评估器
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        初始化评估器
        
        Args:
            model: 模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, dataloader, return_predictions=False):
        """
        评估模型性能
        
        Args:
            dataloader: 数据加载器
            return_predictions: 是否返回预测结果
            
        Returns:
            dict: 评估指标
        """
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        avg_loss = total_loss / num_batches
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'loss': avg_loss
        }
        
        if return_predictions:
            return metrics, all_predictions, all_labels
        
        return metrics
    
    def print_metrics(self, metrics, model_name="模型"):
        """
        打印评估指标
        
        Args:
            metrics: 指标字典
            model_name: 模型名称
        """
        print(f"\n{'='*50}")
        print(f"{model_name} 评估结果:")
        print(f"{'='*50}")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"损失 (Loss): {metrics['loss']:.4f}")
        print(f"{'='*50}\n")


def compare_models(results_dict, save_path='results/comparison.png'):
    """
    可视化比较多个模型的性能
    
    Args:
        results_dict: 字典 {model_name: metrics_dict}
        save_path: 图表保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备数据
    model_names = list(results_dict.keys())
    metrics_names = ['accuracy', 'f1', 'precision', 'recall']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        values = [results_dict[name].get(metric_name, 0) for name in model_names]
        
        bars = ax.bar(range(len(model_names)), values, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} 对比', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存到: {save_path}")
    plt.close()


def compare_model_sizes(results_dict, save_path='results/size_comparison.png'):
    """
    可视化比较模型大小
    
    Args:
        results_dict: 字典 {model_name: {'size_mb': size, 'params': params}}
        save_path: 图表保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_names = list(results_dict.keys())
    sizes = [results_dict[name].get('size_mb', 0) for name in model_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(model_names)), sizes, alpha=0.7, color='coral', edgecolor='darkred')
    
    # 添加数值标签
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{size:.2f} MB',
               ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('大小 (MB)', fontsize=12)
    ax.set_title('模型大小对比', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"大小对比图表已保存到: {save_path}")
    plt.close()


def create_ablation_table(ablation_results, save_path='results/ablation_table.json'):
    """
    创建消融实验表格
    
    Args:
        ablation_results: 消融实验结果列表
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    
    print(f"消融实验结果已保存到: {save_path}")
    
    # 打印表格
    print(f"\n{'='*80}")
    print("消融实验结果")
    print(f"{'='*80}")
    print(f"{'实验':<20} {'准确率':<10} {'F1':<10} {'大小(MB)':<12} {'压缩率':<10}")
    print(f"{'-'*80}")
    
    for result in ablation_results:
        print(f"{result['name']:<20} "
              f"{result['accuracy']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['size_mb']:<12.2f} "
              f"{result.get('compression_ratio', 1.0):<10.2f}")
    
    print(f"{'='*80}\n")


def plot_accuracy_vs_compression(ablation_results, save_path='results/tradeoff.png'):
    """
    绘制准确率 vs 压缩率的权衡曲线
    
    Args:
        ablation_results: 消融实验结果列表
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r['name'] for r in ablation_results]
    accuracies = [r['accuracy'] for r in ablation_results]
    compression_ratios = [r.get('compression_ratio', 1.0) for r in ablation_results]
    
    # 绘制散点图
    scatter = ax.scatter(compression_ratios, accuracies, s=200, alpha=0.6, c=range(len(names)), cmap='viridis')
    
    # 添加标签
    for i, name in enumerate(names):
        ax.annotate(name, (compression_ratios[i], accuracies[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('压缩率 (Compression Ratio)', fontsize=12)
    ax.set_ylabel('准确率 (Accuracy)', fontsize=12)
    ax.set_title('准确率 vs 压缩率权衡', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"权衡曲线已保存到: {save_path}")
    plt.close()


def save_experiment_results(results, save_path='results/experiment_results.json'):
    """
    保存实验结果
    
    Args:
        results: 实验结果字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"实验结果已保存到: {save_path}")
