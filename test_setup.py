"""
快速测试脚本
验证环境配置和基本功能
"""

import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def test_pytorch():
    """测试PyTorch安装"""
    print("测试PyTorch...")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
    print("  ✓ PyTorch正常\n")

def test_transformers():
    """测试transformers安装"""
    print("测试transformers...")
    import transformers
    print(f"  transformers版本: {transformers.__version__}")
    print("  ✓ transformers正常\n")

def test_model_loading():
    """测试模型加载（如果已下载）"""
    print("测试模型加载...")
    model_path = './models/bert-base-uncased'
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        print("  ✓ 本地模型加载成功")
        
        # 测试推理
        text = "This is a great movie!"
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        print("  ✓ 模型推理正常")
        print(f"  输出logits形状: {outputs.logits.shape}\n")
        
    except Exception as e:
        print(f"  ⚠ 模型未下载或加载失败")
        print(f"  错误: {str(e)}")
        print("  请先运行: python download_resources.py\n")

def test_dataset():
    """测试数据集加载（如果已下载）"""
    print("测试数据集加载...")
    dataset_path = './data/sst2'
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        print(f"  ✓ 数据集加载成功")
        print(f"  训练集样本数: {len(dataset['train'])}")
        print(f"  验证集样本数: {len(dataset['validation'])}\n")
        
    except Exception as e:
        print(f"  ⚠ 数据集未下载或加载失败")
        print(f"  错误: {str(e)}")
        print("  请先运行: python download_resources.py\n")

def test_src_modules():
    """测试源代码模块"""
    print("测试源代码模块...")
    
    try:
        from src import utils, model, quantization, pruning, train, evaluate
        print("  ✓ 所有模块导入成功\n")
    except Exception as e:
        print(f"  ✗ 模块导入失败: {str(e)}\n")

def main():
    print("\n" + "="*60)
    print("BERT模型压缩项目 - 环境测试")
    print("="*60 + "\n")
    
    try:
        test_pytorch()
        test_transformers()
        test_src_modules()
        test_model_loading()
        test_dataset()
        
        print("="*60)
        print("环境测试完成！")
        print("="*60)
        print("\n如果模型和数据集未下载，请运行：")
        print("  python download_resources.py")
        print("\n然后可以运行实验：")
        print("  python run_experiments.py --use_subset  # 快速测试")
        print("  python run_experiments.py              # 完整实验\n")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        print("请检查依赖安装: pip install -r requirements.txt\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
