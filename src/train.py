"""
训练模块
提供模型训练和微调功能
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from src.utils import AverageMeter


class BERTTrainer:
    """
    BERT模型训练器
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        初始化训练器
        
        Args:
            model: BERT模型
            tokenizer: 分词器
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def train(self, train_dataloader, val_dataloader=None, 
              num_epochs=3, learning_rate=2e-5, 
              warmup_ratio=0.1, save_dir='models/finetuned'):
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            warmup_ratio: 预热比例
            save_dir: 模型保存目录
            
        Returns:
            dict: 训练历史
        """
        print(f"\n开始训练...")
        print(f"训练样本数: {len(train_dataloader.dataset)}")
        print(f"批次大小: {train_dataloader.batch_size}")
        print(f"训练轮数: {num_epochs}")
        print(f"学习率: {learning_rate}\n")
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        num_training_steps = len(train_dataloader) * num_epochs
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self._train_epoch(train_dataloader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            if val_dataloader is not None:
                val_loss, val_accuracy = self._validate_epoch(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f"验证损失: {val_loss:.4f}")
                print(f"验证准确率: {val_accuracy:.4f}")
                
                # 保存最佳模型
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model(save_dir)
                    print(f"✓ 保存最佳模型 (准确率: {best_val_accuracy:.4f})")
        
        print("\n训练完成！")
        return history
    
    def _train_epoch(self, dataloader, optimizer, scheduler):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            
        Returns:
            float: 平均损失
        """
        self.model.train()
        loss_meter = AverageMeter()
        
        progress_bar = tqdm(dataloader, desc="训练")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 更新指标
            loss_meter.update(loss.item(), input_ids.size(0))
            progress_bar.set_postfix({'loss': loss_meter.avg})
        
        return loss_meter.avg
    
    def _validate_epoch(self, dataloader):
        """
        验证一个epoch
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            tuple: (平均损失, 准确率)
        """
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="验证"):
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
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                loss_meter.update(loss.item(), input_ids.size(0))
        
        accuracy = correct / total
        return loss_meter.avg, accuracy
    
    def save_model(self, save_dir):
        """
        保存模型
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"模型已保存到: {save_dir}")
    
    def load_model(self, load_dir):
        """
        加载模型
        
        Args:
            load_dir: 加载目录
        """
        from transformers import BertForSequenceClassification, BertTokenizer
        
        self.model = BertForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = BertTokenizer.from_pretrained(load_dir)
        self.model.to(self.device)
        print(f"模型已从 {load_dir} 加载")


def finetune_model(model, tokenizer, train_dataloader, val_dataloader=None,
                   num_epochs=3, learning_rate=2e-5, device='cuda',
                   save_dir='models/finetuned'):
    """
    便捷函数：微调BERT模型
    
    Args:
        model: BERT模型
        tokenizer: 分词器
        train_dataloader: 训练数据
        val_dataloader: 验证数据
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        save_dir: 保存目录
        
    Returns:
        tuple: (model, history)
    """
    trainer = BERTTrainer(model, tokenizer, device=device)
    history = trainer.train(
        train_dataloader,
        val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    return model, history


def progressive_finetune(model, tokenizer, train_dataloader, val_dataloader=None,
                        stages=[(3, 5e-5), (2, 2e-5), (2, 1e-5)],
                        device='cuda', save_dir='models/progressive'):
    """
    渐进式微调：多阶段训练策略
    这是本项目的创新点之一，适用于压缩后的模型恢复
    
    Args:
        model: BERT模型
        tokenizer: 分词器
        train_dataloader: 训练数据
        val_dataloader: 验证数据
        stages: 训练阶段列表 [(epochs, lr), ...]
        device: 设备
        save_dir: 保存目录
        
    Returns:
        tuple: (model, history)
    """
    print("\n开始渐进式微调...")
    print(f"训练阶段数: {len(stages)}")
    
    trainer = BERTTrainer(model, tokenizer, device=device)
    all_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for stage_idx, (epochs, lr) in enumerate(stages):
        print(f"\n{'='*50}")
        print(f"阶段 {stage_idx + 1}/{len(stages)}")
        print(f"轮数: {epochs}, 学习率: {lr}")
        print(f"{'='*50}")
        
        history = trainer.train(
            train_dataloader,
            val_dataloader,
            num_epochs=epochs,
            learning_rate=lr,
            save_dir=os.path.join(save_dir, f'stage_{stage_idx + 1}')
        )
        
        # 合并历史
        all_history['train_loss'].extend(history['train_loss'])
        all_history['val_loss'].extend(history['val_loss'])
        all_history['val_accuracy'].extend(history['val_accuracy'])
    
    print("\n渐进式微调完成！")
    return model, all_history
