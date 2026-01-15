"""
实验运行脚本
运行完整的BERT模型压缩实验，包括消融实验
"""

import os
import sys
import argparse
import torch
import copy
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BertTokenizer

from src.model import create_bert_model, BERTModelManager
from src.quantization import quantize_bert_model, BERTQuantizer
from src.pruning import prune_bert_model, BERTPruner, AdaptivePruner, make_pruning_permanent
from src.train import finetune_model, progressive_finetune
from src.evaluate import (ModelEvaluator, compare_models, compare_model_sizes,
                          create_ablation_table, plot_accuracy_vs_compression,
                          save_experiment_results)
from src.utils import (get_model_size, count_parameters, print_model_info, 
                       set_seed, ensure_dir, measure_inference_time)


class SST2DataLoader:
    """
    SST-2数据加载器
    """
    
    def __init__(self, data_dir='./data/sst2', model_name='bert-base-uncased', batch_size=32, max_length=128):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据集目录
            model_name: 模型名称（用于加载tokenizer）
            batch_size: 批次大小
            max_length: 最大序列长度
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 加载tokenizer
        tokenizer_path = f'./models/{model_name}'
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # 加载数据集
        print(f"从本地加载SST-2数据集: {data_dir}")
        self.dataset = load_from_disk(data_dir)
        
        print(f"数据集加载成功！")
        print(f"  训练集: {len(self.dataset['train'])} 样本")
        print(f"  验证集: {len(self.dataset['validation'])} 样本")
    
    def tokenize_function(self, examples):
        """
        分词函数
        """
        return self.tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
    
    def get_dataloaders(self, use_subset=False, subset_size=1000):
        """
        获取数据加载器
        
        Args:
            use_subset: 是否使用子集（用于快速测试）
            subset_size: 子集大小
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # 分词
        print("对数据集进行分词...")
        tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['sentence', 'idx']
        )
        
        # 设置格式
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # 重命名label为labels（PyTorch约定）
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        
        # 使用子集（如果需要）
        if use_subset:
            print(f"使用子集进行快速测试 (训练: {subset_size}, 验证: {subset_size//5})")
            train_dataset = tokenized_dataset['train'].select(range(min(subset_size, len(tokenized_dataset['train']))))
            val_dataset = tokenized_dataset['validation'].select(range(min(subset_size//5, len(tokenized_dataset['validation']))))
        else:
            train_dataset = tokenized_dataset['train']
            val_dataset = tokenized_dataset['validation']
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"DataLoader创建成功！")
        return train_loader, val_loader


def run_baseline_experiment(model, tokenizer, train_loader, val_loader, device='cuda'):
    """
    运行基线实验：原始BERT模型
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验1: 基线模型 (原始BERT)")
    print("="*80)
    
    # 微调
    print("\n微调基线模型...")
    model, history = finetune_model(
        model, tokenizer, train_loader, val_loader,
        num_epochs=3, learning_rate=2e-5, device=device,
        save_dir='./models/baseline'
    )
    
    # 评估
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "基线模型")
    
    # 模型信息
    params = count_parameters(model)
    size = get_model_size(model)
    
    results = {
        'name': '基线-BERT',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': 1.0
    }
    
    print_model_info(model, "基线模型")
    
    return results, model


def run_uniform_pruning_experiment(baseline_model, tokenizer, train_loader, val_loader, device='cuda', baseline_size=None):
    """
    实验2: 统一50%剪枝
    对所有层使用统一的50%稀疏度进行剪枝
    从实验1的微调基线模型开始
    
    Args:
        baseline_model: 实验1的微调基线模型
        baseline_size: 基线模型大小（MB），用于计算压缩率
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验2: 统一50%剪枝")
    print("="*80)
    
    # 创建模型副本（从微调基线开始）
    model = copy.deepcopy(baseline_model)
    model.to(device)
    
    # 应用统一剪枝（所有层50%稀疏度）
    print("\n应用统一50%剪枝...")
    pruner = BERTPruner(model)
    pruner.apply_magnitude_pruning(sparsity_per_layer=0.5)
    
    # 标准微调
    print("\n微调剪枝后的模型...")
    model, history = finetune_model(
        model, tokenizer, train_loader, val_loader,
        num_epochs=2, learning_rate=1e-5, device=device,
        save_dir='./models/uniform_pruning_50'
    )
    
    # 在最终评估前将剪枝永久化
    make_pruning_permanent(model)
    
    # 评估
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "统一50%剪枝模型")
    
    # 模型信息
    params = count_parameters(model)
    size = get_model_size(model)
    if baseline_size is None:
        baseline_size = get_model_size(baseline_model)
    
    results = {
        'name': '统一50%剪枝',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': baseline_size / size if size > 0 else 1.0
    }
    
    print_model_info(model, "统一50%剪枝模型")
    
    return results, model


def run_layer_adaptive_experiment(baseline_model, tokenizer, train_loader, val_loader, device='cuda', baseline_size=None):
    """
    实验3: 层级自适应压缩
    不同层使用不同的稀疏度（浅层30%，深层70%）
    从实验1的微调基线模型开始
    
    Args:
        baseline_model: 实验1的微调基线模型
        baseline_size: 基线模型大小（MB），用于计算压缩率
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验3: 层级自适应压缩")
    print("="*80)
    
    # 创建模型副本（从微调基线开始）
    model = copy.deepcopy(baseline_model)
    model.to(device)
    
    # 应用层级自适应剪枝
    print("\n应用层级自适应剪枝...")
    adaptive_pruner = AdaptivePruner(model)
    adaptive_pruner.apply_layer_adaptive_pruning(base_sparsity=0.5)
    
    # 标准微调
    print("\n微调剪枝后的模型...")
    model, history = finetune_model(
        model, tokenizer, train_loader, val_loader,
        num_epochs=2, learning_rate=1e-5, device=device,
        save_dir='./models/layer_adaptive'
    )
    
    # 在最终评估前将剪枝永久化
    # make_pruning_permanent already imported at top
    make_pruning_permanent(model)
    
    # 评估
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "层级自适应压缩模型")
    
    # 模型信息
    params = count_parameters(model)
    size = get_model_size(model)
    if baseline_size is None:
        baseline_size = get_model_size(baseline_model)
    
    results = {
        'name': '层级自适应压缩',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': baseline_size / size if size > 0 else 1.0
    }
    
    print_model_info(model, "层级自适应压缩模型")
    
    return results, model


def run_adaptive_importance_experiment(exp3_model, tokenizer, train_loader, val_loader, device='cuda', baseline_size=None):
    """
    实验4: 层级自适应压缩 + 改进的重要性评估
    在实验3（层级自适应）基础上，添加基于梯度的重要性评估进行注意力头剪枝
    
    Args:
        exp3_model: 实验3的模型（已应用层级自适应剪枝）
        baseline_size: 基线模型大小（MB），用于计算压缩率
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验4: 层级自适应 + 改进的重要性评估")
    print("="*80)
    
    # 创建模型副本（从实验3开始）
    model = copy.deepcopy(exp3_model)
    model.to(device)
    
    # 步骤1: 使用改进的重要性评估进行注意力头剪枝（创新2）
    print("\n步骤1: 使用改进的重要性评估剪枝注意力头...")
    pruner = BERTPruner(model)
    pruner.compute_head_importance(train_loader, device=device, num_batches=30)
    heads_to_prune = pruner.get_heads_to_prune(prune_ratio=0.3)  # 剪枝30%的头
    pruner.prune_attention_heads(heads_to_prune)
    
    # 步骤2: 微调模型以恢复性能
    print("\n步骤2: 微调模型...")
    model, history = finetune_model(
        model, tokenizer, train_loader, val_loader,
        num_epochs=2, learning_rate=1e-5, device=device,
        save_dir='./models/adaptive_importance'
    )
    
    # 在最终评估前将剪枝永久化
    # make_pruning_permanent already imported at top
    make_pruning_permanent(model)
    
    # 评估
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "层级自适应+重要性评估模型")
    
    # 模型信息
    params = count_parameters(model)
    size = get_model_size(model)
    # baseline_size passed as parameter
    
    results = {
        'name': '层级自适应+重要性评估',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': baseline_size / size if baseline_size and size > 0 else 1.0
    }
    
    print_model_info(model, "层级自适应+重要性评估模型")
    
    return results, model


def run_adaptive_importance_progressive_experiment(exp4_model, tokenizer, train_loader, val_loader, device='cuda', baseline_size=None):
    """
    实验5: 层级自适应 + 改进的重要性评估 + 渐进式恢复训练
    在实验4基础上，添加渐进式多阶段微调（创新3）
    
    Args:
        exp4_model: 实验4的模型（已应用层级自适应+重要性评估）
        baseline_size: 基线模型大小（MB），用于计算压缩率
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验5: 层级自适应 + 重要性评估 + 渐进式恢复训练")
    print("="*80)
    
    # 创建模型副本（从实验4开始）
    model = copy.deepcopy(exp4_model)
    model.to(device)
    
    # 应用渐进式恢复训练（创新3）
    print("\n应用渐进式恢复训练...")
    model, _ = progressive_finetune(
        model, tokenizer, train_loader, val_loader,
        stages=[(2, 2e-5), (2, 1e-5)],
        device=device,
        save_dir='./models/adaptive_importance_progressive'
    )
    
    # 在最终评估前将剪枝永久化
    # make_pruning_permanent already imported at top
    make_pruning_permanent(model)
    
    # 评估
    evaluator = ModelEvaluator(model, tokenizer, device=device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "层级自适应+重要性评估+渐进式训练模型")
    
    # 模型信息
    params = count_parameters(model)
    size = get_model_size(model)
    # baseline_size passed as parameter
    
    results = {
        'name': '层级自适应+重要性评估+渐进式训练',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': baseline_size / size if baseline_size and size > 0 else 1.0
    }
    
    print_model_info(model, "层级自适应+重要性评估+渐进式训练模型")
    
    return results, model


def run_all_innovations_experiment(exp5_model, tokenizer, train_loader, val_loader, device='cuda', baseline_size=None):
    """
    实验6: 所有创新方法组合 + 动态量化
    在实验5基础上，添加动态量化（创新4）
    
    Args:
        exp5_model: 实验5的模型（已应用层级自适应+重要性评估+渐进式训练）
        baseline_size: 基线模型大小（MB），用于计算压缩率
    
    Returns:
        dict: 实验结果
    """
    print("\n" + "="*80)
    print("实验6: 所有创新方法 + 动态量化")
    print("="*80)
    
    # 创建模型副本（从实验5开始）
    model = copy.deepcopy(exp5_model)
    model.to(device)
    
    # 在量化前将剪枝永久化
    make_pruning_permanent(model)
    
    # 应用动态量化（创新4）
    print("\n应用动态量化...")
    quantizer = BERTQuantizer(model)
    quantized_model = quantizer.apply_dynamic_quantization()
    
    # 评估（量化模型在CPU上评估）
    print("\n将量化模型移至CPU进行评估...")
    quantized_model.to('cpu')
    evaluator = ModelEvaluator(quantized_model, tokenizer, device='cpu')
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics, "所有创新方法+量化模型 (在 CPU 上评估)")
    
    # 模型信息
    params = count_parameters(quantized_model)
    size = get_model_size(quantized_model)
    # baseline_size passed as parameter
    
    results = {
        'name': '所有创新方法+量化',
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'loss': metrics['loss'],
        'size_mb': size,
        'params': params['total'],
        'non_zero_params': params['non_zero'],
        'sparsity': params['sparsity'],
        'compression_ratio': baseline_size / size if baseline_size and size > 0 else 1.0
    }
    
    print_model_info(quantized_model, "所有创新方法+量化模型")
    
    # 保存最终模型
    torch.save(quantized_model.state_dict(), './models/all_innovations/model.pth')
    
    return results, quantized_model


def main():
    """
    主函数：运行所有实验
    """
    parser = argparse.ArgumentParser(description='BERT模型压缩实验 - 消融实验')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'baseline', 'uniform', 'layer_adaptive', 'adaptive_importance', 
                                'adaptive_progressive', 'all_innovations'],
                       help='实验模式')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--use_subset', action='store_true',
                       help='使用数据子集（快速测试）')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BERT模型压缩实验 - 消融实验")
    print("="*80)
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"实验模式: {args.mode}")
    print("="*80 + "\n")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建结果目录
    ensure_dir('./results')
    ensure_dir('./models/baseline')
    ensure_dir('./models/uniform_pruning_50')
    ensure_dir('./models/layer_adaptive')
    ensure_dir('./models/adaptive_importance')
    ensure_dir('./models/adaptive_importance_progressive')
    ensure_dir('./models/all_innovations')
    
    # 加载数据
    print("加载数据...")
    data_loader = SST2DataLoader(batch_size=args.batch_size)
    train_loader, val_loader = data_loader.get_dataloaders(
        use_subset=args.use_subset,
        subset_size=1000 if args.use_subset else None
    )
    
    # 加载基础模型
    print("\n加载BERT基础模型...")
    try:
        model, tokenizer, config = create_bert_model(num_labels=2, from_local=True)
    except FileNotFoundError as e:
        print(f"\n错误: 无法加载模型或数据集")
        print(f"详细信息: {str(e)}")
        print("\n请先在联网环境运行以下命令下载资源:")
        print("  python download_resources.py")
        print("\n然后将整个项目文件夹复制到离线环境运行。")
        sys.exit(1)
    
    # 运行实验
    all_results = []
    baseline_size = None
    
    # 实验1: 基线模型
    if args.mode in ['all', 'baseline']:
        baseline_results, baseline_model = run_baseline_experiment(
            copy.deepcopy(model), tokenizer, train_loader, val_loader, args.device
        )
        all_results.append(baseline_results)
        baseline_size = baseline_results['size_mb']
    else:
        baseline_model = model
        baseline_size = get_model_size(baseline_model)
    
    # 实验2: 统一50%剪枝（从基线开始）
    if args.mode in ['all', 'uniform']:
        uniform_results, _ = run_uniform_pruning_experiment(
            baseline_model, tokenizer, train_loader, val_loader, args.device, baseline_size
        )
        all_results.append(uniform_results)
    
    # 实验3: 层级自适应压缩（从基线开始）
    exp3_model = None
    if args.mode in ['all', 'layer_adaptive']:
        layer_adaptive_results, exp3_model = run_layer_adaptive_experiment(
            baseline_model, tokenizer, train_loader, val_loader, args.device, baseline_size
        )
        all_results.append(layer_adaptive_results)
    
    # 实验4: 层级自适应 + 改进的重要性评估（从实验3开始，累积创新）
    exp4_model = None
    if args.mode in ['all', 'adaptive_importance']:
        # 如果没有运行实验3，先运行它
        if exp3_model is None:
            _, exp3_model = run_layer_adaptive_experiment(
                baseline_model, tokenizer, train_loader, val_loader, args.device, baseline_size
            )
        adaptive_importance_results, exp4_model = run_adaptive_importance_experiment(
            exp3_model, tokenizer, train_loader, val_loader, args.device, baseline_size
        )
        all_results.append(adaptive_importance_results)
    
    # 实验5: 层级自适应 + 重要性评估 + 渐进式训练（从实验4开始，累积创新）
    exp5_model = None
    if args.mode in ['all', 'adaptive_progressive']:
        # 如果没有运行实验4，先运行它
        if exp4_model is None:
            if exp3_model is None:
                _, exp3_model = run_layer_adaptive_experiment(
                    baseline_model, tokenizer, train_loader, val_loader, args.device, baseline_size
                )
            _, exp4_model = run_adaptive_importance_experiment(
                exp3_model, tokenizer, train_loader, val_loader, args.device, baseline_size
            )
        adaptive_progressive_results, exp5_model = run_adaptive_importance_progressive_experiment(
            exp4_model, tokenizer, train_loader, val_loader, args.device, baseline_size
        )
        all_results.append(adaptive_progressive_results)
    
    # 实验6: 所有创新方法 + 动态量化（从实验5开始，累积创新）
    if args.mode in ['all', 'all_innovations']:
        # 如果没有运行实验5，先运行它
        if exp5_model is None:
            if exp4_model is None:
                if exp3_model is None:
                    _, exp3_model = run_layer_adaptive_experiment(
                        baseline_model, tokenizer, train_loader, val_loader, args.device, baseline_size
                    )
                _, exp4_model = run_adaptive_importance_experiment(
                    exp3_model, tokenizer, train_loader, val_loader, args.device, baseline_size
                )
            _, exp5_model = run_adaptive_importance_progressive_experiment(
                exp4_model, tokenizer, train_loader, val_loader, args.device, baseline_size
            )
        all_innovations_results, _ = run_all_innovations_experiment(
            exp5_model, tokenizer, train_loader, val_loader, args.device, baseline_size
        )
        all_results.append(all_innovations_results)
    
    # 生成报告
    print("\n" + "="*80)
    print("生成实验报告")
    print("="*80)
    
    # 消融实验表格
    create_ablation_table(all_results, save_path='./results/ablation_results.json')
    
    # 可视化
    if len(all_results) > 1:
        # 性能对比
        metrics_dict = {r['name']: r for r in all_results}
        compare_models(metrics_dict, save_path='./results/performance_comparison.png')
        
        # 大小对比
        size_dict = {r['name']: {'size_mb': r['size_mb'], 'params': r['params']} 
                     for r in all_results}
        compare_model_sizes(size_dict, save_path='./results/size_comparison.png')
        
        # 权衡曲线
        plot_accuracy_vs_compression(all_results, save_path='./results/accuracy_vs_compression.png')
    
    # 保存所有结果
    save_experiment_results(
        {'experiments': all_results, 'config': vars(args)},
        save_path='./results/all_results.json'
    )
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n结果已保存到 ./results/ 目录")
    print("  - ablation_results.json: 消融实验详细结果")
    print("  - performance_comparison.png: 性能对比图")
    print("  - size_comparison.png: 模型大小对比图")
    print("  - accuracy_vs_compression.png: 准确率-压缩率权衡曲线")
    print("  - all_results.json: 完整实验结果\n")


if __name__ == '__main__':
    main()
