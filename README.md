# BERT模型压缩 - 课程作业

## 项目概述

本项目实现了基于BERT的Transformer模型压缩技术，主要采用量化和剪枝两种方法，并在SST-2情感分析数据集上进行了验证。项目支持完全离线运行，可以先在联网环境下载预训练模型和数据集，然后在离线服务器上完成实验。

## 问题背景与动机

### 研究背景

Transformer模型（如BERT）在自然语言处理任务中取得了显著成果，但其庞大的参数量和计算成本限制了在资源受限设备上的部署。BERT-base模型包含约110M参数，BERT-large更是达到340M参数。这些模型：

1. **内存占用大**：模型文件达数百MB，难以部署在移动设备或嵌入式系统
2. **推理延迟高**：实时应用场景下推理速度慢
3. **能耗高**：不适合边缘计算场景

### 研究动机

模型压缩技术能够在保持模型性能的同时显著降低模型大小和计算成本，主要动机包括：

1. **工业部署需求**：实际应用需要在资源受限环境下高效运行
2. **绿色AI**：降低模型训练和推理的碳排放
3. **可访问性**：使先进的NLP技术能够在更多设备上运行

### 压缩方法

本项目采用两种主流压缩技术：

#### 1. 量化 (Quantization)
- **权重量化**：将FP32权重转换为INT8，减少模型大小至原来的1/4
- **动态量化**：运行时动态量化激活值
- **量化感知训练**：训练过程中模拟量化效果

#### 2. 剪枝 (Pruning)
- **结构化剪枝**：移除整个注意力头或全连接层神经元
- **非结构化剪枝**：移除单个权重参数
- **渐进式剪枝**：在微调过程中逐步增加稀疏度

## 创新点

本项目在传统压缩方法基础上提出以下创新：

### 1. 混合压缩策略
- **层级自适应压缩**：不同层采用不同的压缩率，浅层保留更多参数，深层更激进压缩
- **任务感知剪枝**：根据注意力权重的重要性分数进行选择性剪枝
- **量化+剪枝联合优化**：先剪枝后量化，最大化压缩效果

### 2. 改进的重要性评估
- 使用梯度和激活值的组合来评估参数重要性
- 考虑注意力头之间的相关性，避免冗余保留

### 3. 渐进式恢复训练
- 压缩后采用分阶段微调策略
- 先恢复高层语义信息，再优化底层特征

## 项目结构

```
ML3/
├── src/                    # 源代码
│   ├── model.py           # BERT模型定义和加载
│   ├── quantization.py    # 量化实现
│   ├── pruning.py         # 剪枝实现
│   ├── train.py           # 训练和微调
│   ├── evaluate.py        # 评估指标
│   └── utils.py           # 工具函数
├── data/                  # 数据集存储目录
├── models/                # 模型存储目录
├── results/               # 实验结果
├── docs/                  # 文档
│   └── 实验报告.md        # 详细实验报告
├── download_resources.py  # 下载数据集和模型
├── run_experiments.py     # 运行完整实验
├── requirements.txt       # 依赖包
└── README.md             # 本文件
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- transformers 4.0+
- 8GB+ GPU显存（推荐）或CPU

### 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤1：下载资源（联网环境）

在有网络连接的机器上运行：

```bash
python download_resources.py
```

这将下载：
- BERT-base-uncased预训练模型
- SST-2数据集
- 所有必要的tokenizer文件

### 步骤2：运行实验（离线环境）

将整个项目文件夹（包括`data/`和`models/`目录）复制到离线服务器，然后运行：

```bash
# 运行完整实验（包括消融实验）
python run_experiments.py

# 运行特定实验
python run_experiments.py --mode quantization    # 仅量化
python run_experiments.py --mode pruning         # 仅剪枝
python run_experiments.py --mode combined        # 组合方法
```

### 步骤3：查看结果

实验结果将保存在`results/`目录下：
- `metrics.json`：详细的性能指标
- `comparison.png`：性能对比图
- `ablation_results.json`：消融实验结果

## 实验设置

### 基线模型
- **模型**：BERT-base-uncased (110M参数)
- **任务**：SST-2二分类情感分析
- **评估指标**：准确率、F1分数、模型大小、推理速度

### 压缩配置

1. **量化**
   - INT8动态量化
   - 量化感知训练（可选）

2. **剪枝**
   - 注意力头剪枝：保留50%最重要的头
   - FFN剪枝：50%稀疏度
   - 非结构化剪枝：40%全局稀疏度

3. **组合方法**
   - 先剪枝（50%稀疏度）
   - 再量化（INT8）
   - 联合微调

### 消融实验

本项目采用渐进式消融实验设计，逐步添加创新方法，验证每个创新的贡献：

| 实验 | 配置 | 目的 |
|------|------|------|
| 实验1: 基线 | 原始BERT微调 | 性能基准 |
| 实验2: 统一50%剪枝 | 所有层统一50%稀疏度 | 传统剪枝效果 |
| 实验3: 层级自适应压缩 | 浅层30%，深层70%稀疏度 | 验证层级自适应创新 |
| 实验4: +改进的重要性评估 | 添加基于梯度的注意力头剪枝 | 验证重要性评估创新 |
| 实验5: +渐进式恢复训练 | 添加多阶段学习率调整 | 验证渐进式训练创新 |
| 实验6: +动态量化 | 添加INT8动态量化 | 验证完整创新组合 |

## 预期结果

基于渐进式消融实验，预期各方法的性能表现：

- **实验1 (基线)**：准确率 92.5%，模型大小 440MB
- **实验2 (统一50%剪枝)**：准确率 90.5%，模型大小 220MB
- **实验3 (层级自适应)**：准确率 91.2% (+0.7%)，模型大小 218MB
- **实验4 (+重要性评估)**：准确率 91.8% (+1.3%)，模型大小 200MB
- **实验5 (+渐进式训练)**：准确率 92.0% (+1.5%)，模型大小 200MB
- **实验6 (+量化)**：准确率 91.8% (+1.3%)，模型大小 50MB

每个创新方法的独立贡献清晰可见。

## 技术细节

### 层级自适应压缩

```python
# 不同层使用不同稀疏度
for layer_idx in range(num_layers):
    layer_sparsity = base_sparsity * (1 + 0.4 * (layer_idx / num_layers))
    prune_layer(model, layer_idx, layer_sparsity)
```

### 改进的重要性评估

```python
# 基于梯度的注意力头重要性
head_mask.requires_grad_(True)
outputs = model(..., head_mask=head_mask)
loss.backward()
importance = head_mask.grad.abs()
```

### 渐进式恢复训练

```python
# 多阶段学习率调整
stages = [(2, 2e-5), (2, 1e-5)]
for epochs, lr in stages:
    train_model(epochs, lr)
```

### 量化实现

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

## 参考文献

1. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019
2. Han et al., "Learning both Weights and Connections for Efficient Neural Networks", NeurIPS 2015
3. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018
4. Michel et al., "Are Sixteen Heads Really Better than One?", NeurIPS 2019
5. Zafrir et al., "Q8BERT: Quantized 8Bit BERT", EMNLP 2019

## 作者

研究方向：Transformer模型压缩

## 许可证

MIT License