"""
BERT模型加载和管理模块
支持本地加载预训练模型，方便离线使用
"""

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig


class BERTModelManager:
    """
    BERT模型管理器，处理模型的加载、保存和配置
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, local_dir='./models'):
        """
        初始化模型管理器
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类任务标签数
            local_dir: 本地模型存储目录
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.local_dir = local_dir
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def load_model(self, from_local=True):
        """
        加载BERT模型和分词器
        
        Args:
            from_local: 是否从本地加载（True=离线模式）
            
        Returns:
            tuple: (model, tokenizer, config)
        """
        if from_local:
            model_path = os.path.join(self.local_dir, self.model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"本地模型不存在: {model_path}\n"
                    f"请先运行 download_resources.py 下载模型"
                )
            print(f"从本地加载模型: {model_path}")
            self.config = BertConfig.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=self.num_labels
            )
        else:
            print(f"从HuggingFace Hub下载模型: {self.model_name}")
            self.config = BertConfig.from_pretrained(self.model_name)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
        
        print(f"模型加载成功！")
        self._print_model_config()
        
        return self.model, self.tokenizer, self.config
    
    def save_model(self, model, tokenizer, save_path):
        """
        保存模型到本地
        
        Args:
            model: BERT模型
            tokenizer: 分词器
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
    
    def _print_model_config(self):
        """
        打印模型配置信息
        """
        if self.config:
            print(f"\n模型配置:")
            print(f"  - 隐藏层大小: {self.config.hidden_size}")
            print(f"  - 注意力头数: {self.config.num_attention_heads}")
            print(f"  - Transformer层数: {self.config.num_hidden_layers}")
            print(f"  - 中间层大小: {self.config.intermediate_size}")
            print(f"  - 最大序列长度: {self.config.max_position_embeddings}")
            print(f"  - 词汇表大小: {self.config.vocab_size}\n")
    
    def get_model_architecture_info(self):
        """
        获取模型架构详细信息
        
        Returns:
            dict: 架构信息
        """
        if self.model is None or self.config is None:
            raise ValueError("请先加载模型")
        
        info = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'num_hidden_layers': self.config.num_hidden_layers,
            'intermediate_size': self.config.intermediate_size,
            'attention_heads_per_layer': self.config.num_attention_heads,
            'total_attention_heads': self.config.num_attention_heads * self.config.num_hidden_layers
        }
        
        return info


def create_bert_model(num_labels=2, from_local=True, local_dir='./models'):
    """
    便捷函数：创建BERT模型
    
    Args:
        num_labels: 分类标签数
        from_local: 是否从本地加载
        local_dir: 本地目录
        
    Returns:
        tuple: (model, tokenizer, config)
    """
    manager = BERTModelManager(num_labels=num_labels, local_dir=local_dir)
    return manager.load_model(from_local=from_local)


def freeze_bert_layers(model, num_layers_to_freeze=0):
    """
    冻结BERT的前N层，仅微调后面的层
    
    Args:
        model: BERT模型
        num_layers_to_freeze: 要冻结的层数（从0开始）
    """
    if num_layers_to_freeze <= 0:
        return
    
    # 冻结embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    
    # 冻结指定数量的encoder层
    for i in range(num_layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
    
    print(f"已冻结前 {num_layers_to_freeze} 层和embeddings")


def unfreeze_all_layers(model):
    """
    解冻所有层，允许全模型训练
    
    Args:
        model: BERT模型
    """
    for param in model.parameters():
        param.requires_grad = True
    print("所有层已解冻")
