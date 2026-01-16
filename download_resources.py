"""
资源下载脚本
在有网络的环境中运行，下载BERT模型和SST-2数据集
下载后可以在离线环境中使用
"""

import os
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset


def download_bert_model(model_name='bert-base-uncased', save_dir='./models'):
    """
    下载BERT预训练模型
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"\n{'='*60}")
    print(f"下载BERT模型: {model_name}")
    print(f"{'='*60}")
    
    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    # 下载模型
    print("正在下载模型...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # SST-2是二分类任务
    )
    model.save_pretrained(model_save_path)
    print(f"✓ 模型已保存到: {model_save_path}")
    
    # 下载tokenizer
    print("正在下载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_save_path)
    print(f"✓ Tokenizer已保存到: {model_save_path}")
    
    print(f"\n模型下载完成！\n")


def download_sst2_dataset(save_dir='./data'):
    """
    下载SST-2数据集
    
    Args:
        save_dir: 保存目录
    """
    print(f"\n{'='*60}")
    print(f"下载SST-2数据集")
    print(f"{'='*60}")
    
    dataset_save_path = os.path.join(save_dir, 'sst2')
    os.makedirs(dataset_save_path, exist_ok=True)
    
    # 下载数据集
    print("正在下载数据集...")
    dataset = load_dataset('glue', 'sst2')
    
    # 保存到本地
    print("正在保存数据集到本地...")
    dataset.save_to_disk(dataset_save_path)
    
    print(f"✓ 数据集已保存到: {dataset_save_path}")
    
    # 打印数据集信息
    print(f"\n数据集信息:")
    print(f"  训练集: {len(dataset['train'])} 样本")
    print(f"  验证集: {len(dataset['validation'])} 样本")
    print(f"  测试集: {len(dataset['test'])} 样本")
    
    # 显示示例
    print(f"\n数据示例:")
    for i in range(3):
        example = dataset['train'][i]
        print(f"  [{i+1}] 文本: {example['sentence'][:60]}...")
        print(f"      标签: {example['label']} (0=负面, 1=正面)")
    
    print(f"\n数据集下载完成！\n")


def verify_downloads():
    """
    验证下载的文件是否完整
    """
    print(f"\n{'='*60}")
    print("验证下载")
    print(f"{'='*60}\n")
    
    # 验证模型
    model_path = './models/bert-base-uncased'
    if os.path.exists(model_path):
        required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing_files:
            print(f"⚠ 模型文件不完整，缺少: {missing_files}")
        else:
            print(f"✓ BERT模型文件完整")
    else:
        print(f"✗ 未找到BERT模型目录")
    
    # 验证数据集
    dataset_path = './data/sst2'
    if os.path.exists(dataset_path):
        required_dirs = ['train', 'validation', 'test']
        missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(dataset_path, d))]
        
        if missing_dirs:
            print(f"⚠ 数据集不完整，缺少: {missing_dirs}")
        else:
            print(f"✓ SST-2数据集完整")
    else:
        print(f"✗ 未找到SST-2数据集目录")
    
    print(f"\n{'='*60}")
    print("验证完成！")
    print(f"{'='*60}\n")


def main():
    """
    主函数：下载所有资源
    """
    print("\n" + "="*60)
    print("BERT模型压缩项目 - 资源下载工具")
    print("="*60)
    print("\n此脚本将下载以下资源：")
    print("  1. BERT-base-uncased 预训练模型")
    print("  2. SST-2 情感分析数据集")
    print("\n下载后，您可以将整个项目文件夹复制到离线环境使用。")
    print("\n" + "="*60 + "\n")
    
    try:
        # 下载模型
        download_bert_model()
        
        # 下载数据集
        download_sst2_dataset()
        
        # 验证下载
        verify_downloads()
        
        print("\n" + "="*60)
        print("所有资源下载完成！")
        print("="*60)
        print("\n接下来您可以：")
        print("  1. 将整个项目文件夹复制到离线服务器")
        print("  2. 运行: python run_experiments.py")
        print("  3. 查看结果: results/ 目录\n")
        
    except Exception as e:
        print(f"\n✗ 下载过程中出现错误: {str(e)}")
        print("请检查网络连接并重试。\n")
        raise


if __name__ == '__main__':
    main()
