"""
使用示例
展示如何使用项目的各个模块
"""

import torch
from src import (
    create_bert_model,
    quantize_bert_model,
    prune_bert_model,
    get_model_size,
    count_parameters,
    print_model_info
)


def example_1_load_model():
    """
    示例1: 加载BERT模型
    """
    print("\n" + "="*60)
    print("示例1: 加载BERT模型")
    print("="*60)
    
    # 从本地加载（离线模式）
    model, tokenizer, config = create_bert_model(
        num_labels=2,
        from_local=True,
        local_dir='./models'
    )
    
    print_model_info(model, "BERT-base")
    
    # 简单推理测试
    text = "This movie is fantastic!"
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
    
    print(f"输入文本: {text}")
    print(f"预测结果: {prediction.item()} ({'正面' if prediction.item() == 1 else '负面'})")
    
    return model, tokenizer


def example_2_quantization(model):
    """
    示例2: 模型量化
    """
    print("\n" + "="*60)
    print("示例2: 模型量化")
    print("="*60)
    
    # 获取原始模型大小
    original_size = get_model_size(model)
    print(f"原始模型大小: {original_size:.2f} MB")
    
    # 应用动态量化
    quantized_model = quantize_bert_model(model, method='dynamic')
    
    # 获取量化后模型大小
    quantized_size = get_model_size(quantized_model)
    print(f"量化后模型大小: {quantized_size:.2f} MB")
    print(f"压缩率: {original_size / quantized_size:.2f}x")
    print(f"大小减少: {(1 - quantized_size / original_size) * 100:.1f}%")
    
    return quantized_model


def example_3_pruning(model, tokenizer):
    """
    示例3: 模型剪枝
    """
    print("\n" + "="*60)
    print("示例3: 模型剪枝（非结构化）")
    print("="*60)
    
    import copy
    model_copy = copy.deepcopy(model)
    
    # 获取原始参数信息
    original_params = count_parameters(model_copy)
    print(f"原始参数量: {original_params['total']:,}")
    print(f"原始稀疏度: {original_params['sparsity']:.2%}")
    
    # 应用非结构化剪枝（不需要dataloader）
    from src.pruning import BERTPruner
    pruner = BERTPruner(model_copy)
    pruner.apply_unstructured_pruning(sparsity=0.4)
    
    # 获取剪枝后参数信息
    pruned_params = count_parameters(model_copy)
    print(f"\n剪枝后参数量: {pruned_params['total']:,}")
    print(f"非零参数: {pruned_params['non_zero']:,}")
    print(f"剪枝后稀疏度: {pruned_params['sparsity']:.2%}")
    
    return model_copy


def example_4_combined_compression(model):
    """
    示例4: 组合压缩（先剪枝后量化）
    """
    print("\n" + "="*60)
    print("示例4: 组合压缩（剪枝 + 量化）")
    print("="*60)
    
    import copy
    model_copy = copy.deepcopy(model)
    
    # 原始模型大小
    original_size = get_model_size(model)
    original_params = count_parameters(model)
    
    print(f"原始模型:")
    print(f"  大小: {original_size:.2f} MB")
    print(f"  参数: {original_params['total']:,}")
    
    # 步骤1: 剪枝
    print("\n步骤1: 应用剪枝...")
    from src.pruning import BERTPruner
    pruner = BERTPruner(model_copy)
    pruner.apply_unstructured_pruning(sparsity=0.5)
    
    pruned_size = get_model_size(model_copy)
    pruned_params = count_parameters(model_copy)
    print(f"  剪枝后大小: {pruned_size:.2f} MB")
    print(f"  剪枝后稀疏度: {pruned_params['sparsity']:.2%}")
    
    # 步骤2: 量化
    print("\n步骤2: 应用量化...")
    quantized_model = quantize_bert_model(model_copy, method='dynamic')
    
    final_size = get_model_size(quantized_model)
    print(f"  最终大小: {final_size:.2f} MB")
    
    # 总结
    print(f"\n总结:")
    print(f"  压缩率: {original_size / final_size:.2f}x")
    print(f"  大小减少: {(1 - final_size / original_size) * 100:.1f}%")
    
    return quantized_model


def example_5_comparison():
    """
    示例5: 对比不同压缩方法
    """
    print("\n" + "="*60)
    print("示例5: 对比不同压缩方法")
    print("="*60)
    
    import copy
    
    # 加载模型
    model, tokenizer, _ = create_bert_model(from_local=True)
    original_size = get_model_size(model)
    
    # 测试不同方法
    results = {}
    
    # 1. 原始模型
    results['原始模型'] = {
        'size': original_size,
        'compression_ratio': 1.0
    }
    
    # 2. 量化
    model_quant = quantize_bert_model(copy.deepcopy(model), method='dynamic')
    quant_size = get_model_size(model_quant)
    results['量化'] = {
        'size': quant_size,
        'compression_ratio': original_size / quant_size
    }
    
    # 3. 剪枝30%
    model_prune_30 = copy.deepcopy(model)
    from src.pruning import BERTPruner
    BERTPruner(model_prune_30).apply_unstructured_pruning(0.3)
    prune_30_size = get_model_size(model_prune_30)
    results['剪枝30%'] = {
        'size': prune_30_size,
        'compression_ratio': original_size / prune_30_size
    }
    
    # 4. 剪枝50%
    model_prune_50 = copy.deepcopy(model)
    BERTPruner(model_prune_50).apply_unstructured_pruning(0.5)
    prune_50_size = get_model_size(model_prune_50)
    results['剪枝50%'] = {
        'size': prune_50_size,
        'compression_ratio': original_size / prune_50_size
    }
    
    # 5. 组合（剪枝50% + 量化）
    model_combined = copy.deepcopy(model)
    BERTPruner(model_combined).apply_unstructured_pruning(0.5)
    model_combined = quantize_bert_model(model_combined, method='dynamic')
    combined_size = get_model_size(model_combined)
    results['剪枝50%+量化'] = {
        'size': combined_size,
        'compression_ratio': original_size / combined_size
    }
    
    # 打印对比表格
    print(f"\n{'方法':<15} {'大小(MB)':<12} {'压缩率':<10} {'大小减少':<10}")
    print("-" * 50)
    
    for method, data in results.items():
        size = data['size']
        ratio = data['compression_ratio']
        reduction = (1 - 1/ratio) * 100 if ratio > 0 else 0
        print(f"{method:<15} {size:<12.2f} {ratio:<10.2f} {reduction:<10.1f}%")
    
    print("-" * 50)


def main():
    """
    运行所有示例
    """
    print("\n" + "="*60)
    print("BERT模型压缩 - 使用示例")
    print("="*60)
    print("\n注意: 运行此脚本前请确保已下载模型和数据集")
    print("运行: python download_resources.py\n")
    
    try:
        # 示例1: 加载模型
        model, tokenizer = example_1_load_model()
        
        # 示例2: 量化
        _ = example_2_quantization(model)
        
        # 示例3: 剪枝
        _ = example_3_pruning(model, tokenizer)
        
        # 示例4: 组合压缩
        _ = example_4_combined_compression(model)
        
        # 示例5: 对比
        example_5_comparison()
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("\n接下来可以运行完整实验:")
        print("  python run_experiments.py --use_subset  # 快速测试")
        print("  python run_experiments.py              # 完整实验\n")
        
    except FileNotFoundError as e:
        print(f"\n错误: {str(e)}")
        print("请先运行: python download_resources.py\n")
    except Exception as e:
        print(f"\n运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
