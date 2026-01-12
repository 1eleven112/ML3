# 快速开始指南

本指南帮助您快速上手BERT模型压缩项目。

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python test_setup.py
```

这将检查：
- PyTorch是否正确安装
- CUDA是否可用（如果有GPU）
- transformers库是否正常
- 源代码模块是否可导入

## 使用流程

### 方案A：联网环境 → 离线环境（推荐）

**在联网机器上：**

```bash
# 1. 克隆项目
git clone https://github.com/1eleven112/ML3.git
cd ML3

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载资源（需要网络）
python download_resources.py
```

这将下载：
- BERT-base-uncased预训练模型（约440MB）
- SST-2数据集（约10MB）

**复制到离线服务器：**

```bash
# 将整个项目文件夹打包
tar -czf ML3.tar.gz ML3/

# 传输到离线服务器
scp ML3.tar.gz user@offline-server:/path/to/destination/

# 在离线服务器上解压
ssh user@offline-server
cd /path/to/destination/
tar -xzf ML3.tar.gz
cd ML3
```

**在离线服务器上运行实验：**

```bash
# 完整实验（需要2-3小时，GPU环境）
python run_experiments.py

# 或者先用小数据集快速测试（5-10分钟）
python run_experiments.py --use_subset

# 只运行特定实验
python run_experiments.py --mode quantization    # 仅量化
python run_experiments.py --mode pruning         # 仅剪枝
python run_experiments.py --mode combined        # 组合方法
```

### 方案B：完全在线环境

```bash
# 1. 克隆项目
git clone https://github.com/1eleven112/ML3.git
cd ML3

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载资源
python download_resources.py

# 4. 运行实验
python run_experiments.py --use_subset  # 快速测试
python run_experiments.py              # 完整实验
```

## 实验参数说明

```bash
python run_experiments.py --help
```

主要参数：
- `--mode`: 实验模式
  - `all`: 运行所有实验（默认）
  - `baseline`: 仅基线模型
  - `quantization`: 仅量化实验
  - `pruning`: 仅剪枝实验
  - `combined`: 仅组合方法
  - `adaptive`: 仅创新方法
- `--device`: 设备选择（`cuda`或`cpu`，自动检测）
- `--batch_size`: 批次大小（默认32）
- `--use_subset`: 使用数据子集快速测试

## 查看结果

实验完成后，结果保存在`results/`目录：

```
results/
├── ablation_results.json           # 消融实验详细数据
├── performance_comparison.png      # 性能对比图
├── size_comparison.png            # 模型大小对比图
├── accuracy_vs_compression.png    # 准确率-压缩率权衡曲线
└── all_results.json               # 完整实验结果
```

查看JSON结果：
```bash
# Linux/Mac
cat results/ablation_results.json | python -m json.tool

# 或使用jq（如果已安装）
cat results/ablation_results.json | jq '.'
```

查看图片：
- 直接在文件管理器中打开PNG文件
- 或使用图片查看器

## 项目结构

```
ML3/
├── src/                    # 源代码
│   ├── model.py           # BERT模型加载
│   ├── quantization.py    # 量化实现
│   ├── pruning.py         # 剪枝实现
│   ├── train.py           # 训练模块
│   ├── evaluate.py        # 评估模块
│   └── utils.py           # 工具函数
├── data/                  # 数据集（下载后）
│   └── sst2/
├── models/                # 模型文件（下载后）
│   ├── bert-base-uncased/
│   ├── baseline/          # 微调后的基线
│   ├── quantized/         # 量化模型
│   ├── pruned_50/         # 50%剪枝模型
│   └── ...
├── results/               # 实验结果
├── docs/                  # 文档
│   └── 实验报告.md
├── download_resources.py  # 下载脚本
├── run_experiments.py     # 实验脚本
├── test_setup.py         # 测试脚本
├── requirements.txt      # 依赖
└── README.md            # 主文档
```

## 常见问题

### Q1: 下载速度慢怎么办？

A: 可以使用国内镜像：
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中设置
# 编辑download_resources.py，添加镜像配置
```

### Q2: 内存不足怎么办？

A: 减小批次大小：
```bash
python run_experiments.py --batch_size 16
# 或更小: --batch_size 8
```

### Q3: 没有GPU可以运行吗？

A: 可以，但会慢很多：
```bash
# 系统会自动检测并使用CPU
python run_experiments.py --device cpu --use_subset
```

### Q4: 如何只运行部分实验？

A: 使用`--mode`参数：
```bash
# 只运行量化实验
python run_experiments.py --mode quantization

# 只运行剪枝实验
python run_experiments.py --mode pruning
```

### Q5: 实验需要多长时间？

A: 取决于硬件：
- **GPU (V100)**: 完整实验约2-3小时
- **GPU (GTX 1080)**: 约4-6小时
- **CPU**: 约8-12小时
- **快速测试** (--use_subset): 10-30分钟

## 进阶使用

### 自定义实验

可以修改`run_experiments.py`中的参数：

```python
# 修改剪枝比例
prune_ratio = 0.6  # 改为60%

# 修改训练轮数
num_epochs = 5  # 改为5轮

# 修改学习率
learning_rate = 3e-5  # 调整学习率
```

### 使用API

项目模块可以作为库使用：

```python
from src import create_bert_model, quantize_bert_model, prune_bert_model

# 加载模型
model, tokenizer, config = create_bert_model(from_local=True)

# 量化
quantized_model = quantize_bert_model(model, method='dynamic')

# 剪枝
pruned_model = prune_bert_model(model, method='heads', prune_ratio=0.5)
```

## 获取帮助

如有问题，请：
1. 查看`docs/实验报告.md`获取详细技术文档
2. 检查`README.md`了解项目背景
3. 查看源代码注释
4. 提交GitHub Issue

## 引用

如果本项目对您有帮助，请引用：

```
@misc{ml3-bert-compression,
  title={BERT Model Compression: Quantization and Pruning},
  author={ML3 Project},
  year={2024},
  url={https://github.com/1eleven112/ML3}
}
```
